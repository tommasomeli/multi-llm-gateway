import os
import yaml
import subprocess
import time
import httpx
import logging
import asyncio
from typing import Dict, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MultiModelGateway")

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.yaml")
processes: List[subprocess.Popen] = []
model_map: Dict[str, int] = {}  # Map model_name -> port

def load_config() -> Dict:
    # 1. Try to load from environment variable CONFIG_JSON
    config_json = os.getenv("CONFIG_JSON")
    if config_json:
        try:
            logger.info("Loading config from CONFIG_JSON environment variable...")
            return json.loads(config_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CONFIG_JSON: {e}")
            # Fallback to file

    # 2. Try to load from file
    if not os.path.exists(CONFIG_PATH):
        # If no file and no env var, we cannot proceed
        if not config_json:
             raise FileNotFoundError(f"Config file not found at {CONFIG_PATH} and CONFIG_JSON env var not set")
    
    logger.info(f"Loading config from file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def start_vllm_instance(model_cfg: Dict):
    name = model_cfg["name"]
    model_id = model_cfg["model_id"]
    gpu_ids = model_cfg.get("gpu_ids", "0")
    port = model_cfg["port"]
    extra_args = model_cfg.get("args", "")

    # Handle wildcard "*" for all GPUs
    if gpu_ids == "*":
        try:
            import torch
            device_count = torch.cuda.device_count()
            gpu_ids = ",".join(map(str, range(device_count)))
            logger.info(f"Wildcard '*' detected for model '{name}'. Using all {device_count} GPUs: {gpu_ids}")
            
            # Auto-inject tensor-parallel-size if not manually specified
            if "--tensor-parallel-size" not in extra_args:
                logger.info(f"Auto-setting --tensor-parallel-size {device_count} for model '{name}'")
                extra_args += f" --tensor-parallel-size {device_count}"
        except Exception as e:
            logger.error(f"Failed to resolve '*' GPUs: {e}. Defaulting to GPU 0.")
            gpu_ids = "0"

    # Convert list of GPUs to string if necessary
    if isinstance(gpu_ids, list):
        gpu_ids = ",".join(map(str, gpu_ids))

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--served-model-name", name,
        "--port", str(port),
        "--host", "127.0.0.1"
    ]
    
    if extra_args:
        cmd.extend(extra_args.split())

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)

    logger.info(f"Starting model '{name}' on port {port} with GPUs {gpu_ids}...")
    proc = subprocess.Popen(cmd, env=env)
    processes.append(proc)
    model_map[name] = port
    
    # Also map the HF model ID to the port for convenience
    model_map[model_id] = port

async def check_health(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url)
            return resp.status_code == 200
        except:
            return False

async def wait_for_models(models: List[Dict]):
    """Wait for all models to be healthy."""
    logger.info("Waiting for models to become ready...")
    start_time = time.time()
    timeout = 600  # 10 minutes timeout
    
    pending_ports = {m["port"] for m in models}
    
    while pending_ports and (time.time() - start_time < timeout):
        ready_ports = set()
        for port in pending_ports:
            if await check_health(port):
                logger.info(f"Model on port {port} is ready.")
                ready_ports.add(port)
        
        pending_ports -= ready_ports
        
        if pending_ports:
            await asyncio.sleep(2)
            
    if pending_ports:
        logger.error(f"Timeout waiting for models on ports: {pending_ports}")
    else:
        logger.info("All models are ready!")

def setup_environment(config: Dict):
    """Setup environment variables based on config."""
    storage_cfg = config.get("storage", {})
    # Support 'models_dir' or fallback to 'cache_dir'
    models_dir = storage_cfg.get("models_dir") or storage_cfg.get("cache_dir")
    
    if models_dir:
        logger.info(f"Setting HF_HOME to {models_dir}")
        os.makedirs(models_dir, exist_ok=True)
        # We set HF_HOME so Hugging Face libraries know where to store/find models
        os.environ["HF_HOME"] = models_dir

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    setup_environment(config)
    
    models = config.get("models", [])
    
    for model in models:
        start_vllm_instance(model)
    
    # Wait for models to load in background to not block startup completely?
    # Or block to ensure readiness? Blocking is safer for container probes.
    await wait_for_models(models)
    
    yield
    
    # Shutdown
    logger.info("Shutting down worker processes...")
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    
    # Give them a chance to shut down gracefully
    for _ in range(10):
        if all(proc.poll() is not None for proc in processes):
            break
        await asyncio.sleep(0.5)

    for proc in processes:
        if proc.poll() is None:
            proc.kill()

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(model_map.keys())}

@app.get("/v1/models")
async def list_models():
    # Aggregate models from all workers or just return config
    # For simplicity, return configured models
    models_list = []
    for model_name in model_map.keys():
        # Avoid duplicates from ID/Name mapping
        if "/" not in model_name: 
            models_list.append({
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "runpod-user"
            })
    return {"object": "list", "data": models_list}

async def proxy_request(request: Request, port: int, path: str):
    url = f"http://127.0.0.1:{port}{path}"
    
    # Extract body if present
    body = await request.body()
    
    async with httpx.AsyncClient() as client:
        # Prepare headers (filter hop-by-hop)
        excluded_headers = {
            "host", "content-length", "transfer-encoding", "connection", 
            "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "upgrade"
        }
        headers = {k: v for k, v in request.headers.items() if k.lower() not in excluded_headers}

        try:
            req = client.build_request(
                request.method,
                url,
                headers=headers,
                content=body,
                timeout=None # Let vLLM handle timeouts
            )
            
            r = await client.send(req, stream=True)
            
            # Filter response headers too
            resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in excluded_headers}

            return StreamingResponse(
                r.aiter_raw(),
                status_code=r.status_code,
                headers=resp_headers,
                background=r.aclose
            )
        except httpx.RequestError as exc:
            raise HTTPException(status_code=502, detail=f"Model worker error: {exc}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    model_name = body.get("model")
    if not model_name or model_name not in model_map:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available: {list(model_map.keys())}")
    
    port = model_map[model_name]
    return await proxy_request(request, port, "/v1/chat/completions")

@app.post("/v1/completions")
async def completions(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON")
        
    model_name = body.get("model")
    if not model_name or model_name not in model_map:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    port = model_map[model_name]
    return await proxy_request(request, port, "/v1/completions")

if __name__ == "__main__":
    import uvicorn
    config = load_config()
    server_cfg = config.get("server", {})
    uvicorn.run(app, host=server_cfg.get("host", "0.0.0.0"), port=server_cfg.get("port", 8000))
