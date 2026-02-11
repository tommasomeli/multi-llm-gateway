#!/bin/bash
set -e

# Script to pre-download models during build (optional)
# Usage: ./builder/setup.sh

echo "Downloading models defined in config.yaml..."

# Python script to download models
python -c "
import yaml
import os
from huggingface_hub import snapshot_download

config_path = 'config.yaml'
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        for model in config.get('models', []):
            print(f'Downloading {model[\"model_id\"]}...')
            snapshot_download(repo_id=model['model_id'])
else:
    print('config.yaml not found, skipping pre-download')
"
