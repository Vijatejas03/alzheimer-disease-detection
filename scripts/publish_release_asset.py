"""
Automated GitHub Release Asset Publisher for ResNet-18 (128 MB).
Creates release v1.0.0 on Vijatejas03/alzheimer-disease-detection and uploads
results/models/resnet18_best.pt directly via GitHub API.
Securely retrieves git credentials in memory without logging or saving secrets.
"""

import os
import sys
import json
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "results" / "models" / "resnet18_best.pt"

def get_github_token() -> str:
    gcm_path = r"C:\Users\vijay\AppData\Local\PortableGit\mingw64\bin\git-credential-manager.exe"
    env = dict(os.environ)
    env["PATH"] = r"C:\Users\vijay\AppData\Local\PortableGit\cmd;" + env.get("PATH", "")
    
    p = subprocess.Popen([gcm_path, "get"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, _ = p.communicate("protocol=https\nhost=github.com\n")
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("Failed to retrieve GitHub credential from Git Credential Manager")

def publish():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    
    file_size = MODEL_PATH.stat().st_size
    print(f"Target model file: {MODEL_PATH.name} ({file_size / (1024*1024):.2f} MB)")
    
    token = get_github_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Alzheimer-Deployment-Publisher"
    }
    
    repo = "Vijatejas03/alzheimer-disease-detection"
    base_api = f"https://api.github.com/repos/{repo}"
    
    # 1. Check if release v1.0.0 exists
    release = None
    try:
        req = urllib.request.Request(f"{base_api}/releases/tags/v1.0.0", headers=headers)
        with urllib.request.urlopen(req) as resp:
            release = json.load(resp)
            print(f"Existing release 'v1.0.0' found (ID: {release['id']})")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print("Release 'v1.0.0' does not exist yet. Creating release...")
        else:
            raise
            
    # 2. Create release if missing
    if release is None:
        payload = {
            "tag_name": "v1.0.0",
            "target_commitish": "main",
            "name": "v1.0.0 - Production Neural Model Artifacts",
            "body": "Official Release v1.0.0 providing model weights for the Alzheimer Disease Detection & Explainability System.",
            "draft": False,
            "prerelease": False
        }
        req = urllib.request.Request(
            f"{base_api}/releases",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            release = json.load(resp)
            print(f"Release 'v1.0.0' created successfully (ID: {release['id']})")
            
    release_id = release["id"]
    upload_url_template = release.get("upload_url", "")
    # upload_url_template is like https://uploads.github.com/repos/.../releases/123/assets{?name,label}
    upload_url = upload_url_template.split("{")[0]
    
    # 3. Check if resnet18_best.pt already uploaded
    assets = release.get("assets", [])
    existing_asset = next((a for a in assets if a["name"] == "resnet18_best.pt"), None)
    if existing_asset and existing_asset.get("size") == file_size:
        print(f"Asset 'resnet18_best.pt' already exists on release v1.0.0 with matching size ({existing_asset['size']} bytes).")
        print("Download URL:", existing_asset.get("browser_download_url"))
        return existing_asset.get("browser_download_url")
    elif existing_asset:
        print(f"Asset exists with mismatched size ({existing_asset.get('size')} != {file_size}). Deleting to re-upload...")
        del_req = urllib.request.Request(existing_asset["url"], headers=headers, method="DELETE")
        with urllib.request.urlopen(del_req) as resp:
            print("Old asset deleted.")
            
    # 4. Upload resnet18_best.pt
    print(f"\nUploading {MODEL_PATH.name} ({file_size / (1024*1024):.2f} MB) to GitHub Release...")
    with open(MODEL_PATH, "rb") as f:
        file_bytes = f.read()
        
    upload_endpoint = f"{upload_url}?name={MODEL_PATH.name}"
    upload_headers = dict(headers)
    upload_headers["Content-Type"] = "application/octet-stream"
    upload_headers["Content-Length"] = str(file_size)
    
    up_req = urllib.request.Request(upload_endpoint, data=file_bytes, headers=upload_headers, method="POST")
    with urllib.request.urlopen(up_req, timeout=300) as resp:
        asset_info = json.load(resp)
        dl_url = asset_info.get("browser_download_url")
        print(f"Asset uploaded successfully!")
        print(f"Public Download URL: {dl_url}")
        return dl_url

if __name__ == "__main__":
    publish()
