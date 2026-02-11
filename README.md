# Multi-LLM Gateway for RunPod (vLLM + FastAPI)

Deploy multiple vLLM instances on a single GPU server (RunPod), orchestrated by a FastAPI gateway.

## Features

- **Multi-Model**: Run different models on different GPUs (e.g., Llama-3 on GPU 0, Mistral on GPU 1).
- **Single Endpoint**: OpenAI-compatible API (`/v1/chat/completions`) routing to the correct model.
- **Dynamic Config**: Fully configurable via environment variables (perfect for RunPod Templates).

## RunPod Configuration

### 1. Template Setup

- **Image**: `osammot22/multi-llm-gateway:v1`
- **Volume Mount Path**: `/home/models` (Recommended for caching models)
- **Container Disk**: 50GB+
- **Volume Disk**: 100GB+
- **Ports**: `8000` (HTTP)

### 2. Environment Variables

Control everything from the RunPod dashboard without rebuilding the image.

| Variable                 | Description                                                          | Example   |
| ------------------------ | -------------------------------------------------------------------- | --------- |
| `HUGGING_FACE_HUB_TOKEN` | Your HF token for gated models.                                      | `hf_...`  |
| `CONFIG_JSON`            | **(Optional)** Overrides `config.yaml`. JSON string defining models. | See below |

#### Example `CONFIG_JSON`

Paste this minified JSON into the RunPod Env Var to define your setup:

```json
{
  "models": [
    {
      "name": "llama-3",
      "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
      "gpu_ids": [0],
      "port": 8001
    }
  ],
  "storage": { "models_dir": "/home/models" }
}
```

## Massive Models (Multi-GPU)

To run large models like **Qwen/Qwen3-Coder-Next** (80B+ params) that require multiple GPUs (Tensor Parallelism):

1. **Allocate multiple GPUs** in `gpu_ids` (e.g., `[0, 1]` or `"*"` to use all available).
2. **Add `--tensor-parallel-size N`** to `args`. If you use `gpu_ids: "*"`, the system will automatically set `tensor-parallel-size` to the number of available GPUs.

Example `config.yaml` using all available GPUs:

```yaml
models:
  - name: "qwen-coder-80b"
    model_id: "Qwen/Qwen3-Coder-Next"
    gpu_ids: "*"  # Automatically use all GPUs
    port: 8001
    # System will auto-inject --tensor-parallel-size matching GPU count
    args: "--trust-remote-code --max-model-len 8192 --dtype bfloat16"
```

## Local Development

1. **Setup**:
   ```bash
   cp .env.example .env  # Add your HF Token
   pip install -r requirements.txt
   ```
2. **Run**:
   ```bash
   ./start.sh
   ```

## API Usage

**Chat Completion**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
