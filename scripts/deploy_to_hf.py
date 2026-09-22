"""
Automated Deployment Script for Hugging Face Spaces.
Uses huggingface_hub to create and synchronize the Streamlit deployment repository.
Handles large model weights (including ResNet-18 at 128 MB) seamlessly via Hugging Face LFS.
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def deploy(token: str, space_name: str = "alzheimer-disease-detection"):
    api = HfApi(token=token)
    user = api.whoami()
    username = user["name"]
    repo_id = f"{username}/{space_name}"
    
    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Target Space Repository ID: {repo_id}")
    
    # 1. Create Space if it doesn't exist
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space '{repo_id}' already exists. Updating files...")
    except Exception:
        print(f"Creating new Space '{repo_id}' with Streamlit SDK...")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="streamlit",
            token=token,
            private=False
        )
        print(f"Space '{repo_id}' created successfully.")

    # 2. Upload Deployment Files
    # Exclude raw bulk data, git directory, temp caches
    ignore_patterns = [
        "data/dataset/**",
        "data/raw/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "scratch/**",
        ".venv/**",
        "venv/**",
        "reports/screenshots/**"
    ]
    
    print("\nUploading deployment package to Hugging Face Spaces (including all 3 model checkpoints)...")
    upload_info = api.upload_folder(
        folder_path=str(PROJECT_ROOT),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=ignore_patterns,
        commit_message="Deploy Alzheimer XAI Platform v1.0",
        token=token
    )
    print("Files uploaded successfully.")
    
    public_url = f"https://huggingface.co/spaces/{repo_id}"
    embed_url = f"https://{username}-{space_name.replace('_', '-')}.hf.space"
    print(f"\n==========================================")
    print(f"PUBLIC SPACE URL: {public_url}")
    print(f"DIRECT STREAMLIT URL: {embed_url}")
    print(f"==========================================")
    
    # 3. Monitor Build Status
    print("\nMonitoring Space build status...")
    max_wait_seconds = 300
    start_time = time.time()
    while time.time() - start_time < max_wait_seconds:
        runtime = api.get_space_runtime(repo_id=repo_id)
        stage = runtime.stage
        print(f"Current Space Stage: {stage}")
        if stage == "RUNNING":
            print("\nSpace is LIVE and RUNNING!")
            return public_url, embed_url, True
        elif stage in ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"):
            print(f"\nDeployment encountered an error stage: {stage}")
            return public_url, embed_url, False
        time.sleep(10)
        
    print("\nBuild monitoring timed out. The Space may still be building.")
    return public_url, embed_url, False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/deploy_to_hf.py <HF_ACCESS_TOKEN> [SPACE_NAME]")
        sys.exit(1)
    tok = sys.argv[1]
    s_name = sys.argv[2] if len(sys.argv) > 2 else "alzheimer-disease-detection"
    deploy(tok, s_name)
