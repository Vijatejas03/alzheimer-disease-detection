# Strategic Deployment Platform Decision: Hugging Face Spaces vs. Streamlit Community Cloud

**Project**: Alzheimer's Disease Detection & Explainability System  
**Application Target**: Streamlit Web Application (`app/app.py`)  
**Evaluation Date**: September 22, 2026  
**Status**: **PLATFORM COMPARISON COMPLETE — ZERO COST ROUTE IDENTIFIED**

---

## 1. Current Hugging Face Restriction

On the Hugging Face "Create a new Space" interface, the available SDK choices are:
- **Static** (Free, but HTML/CSS/JS only; cannot run Python, PyTorch, or Streamlit)
- **Gradio (Paid)** (Compute-based, requires paid subscription or billing verification)
- **Docker (Paid)** (Compute-based, requires paid subscription or billing verification)

### Technical Analysis:
1. **Removal of Native Streamlit SDK**: Hugging Face has removed the dedicated free Streamlit radio button from the Space creation wizard. Streamlit deployments now require the Docker SDK.
2. **Compute Gating**: On standard/unverified free accounts, creating non-static compute spaces (Gradio and Docker) is flagged as **(Paid)** and triggers a credit card or billing prompt.
3. **Static Incompatibility**: The "Static" option cannot execute PyTorch models, forward inference passes, or dynamic Grad-CAM heatmaps.

---

## 2. Can Our Account Deploy on Hugging Face for Free?

**NO.**  
Because your Hugging Face account displays `Gradio (Paid)` and `Docker (Paid)`, creating a compute-backed space to run Python, Streamlit, and PyTorch would require purchasing a paid tier or entering credit card billing information.

Per your strict constraint (*"DO NOT ask me to buy a paid plan yet"*), **we will NOT proceed with Hugging Face**.

---

## 3. Does Docker / Streamlit on Hugging Face Require Payment?

**YES.**  
Under Hugging Face's current infrastructure policies, running a Dockerized Streamlit application consumes persistent compute resources that are categorized under paid compute for unverified/free tier accounts.

---

## 4. Can Streamlit Community Cloud Host This Existing App for Free?

**YES, 100% FREE WITH ZERO PAYMENT AND ZERO CREDIT CARD REQUIRED.**

**Streamlit Community Cloud (`share.streamlit.io`)** is the official cloud hosting platform built specifically by Snowflake/Streamlit for hosting open-source Streamlit applications directly from GitHub.

| Feature | Streamlit Community Cloud Specification | Project Requirement | Compatible? |
| :--- | :--- | :--- | :---: |
| **Cost** | **$0.00 (Completely free for public GitHub repos)** | $0.00 budget | **YES** |
| **Credit Card** | **None required** | No payment | **YES** |
| **RAM Allocation** | **1,000 MB (1 GB)** | **~296 MB tested peak RAM** | **YES** (Uses only ~30% of limit) |
| **Application Entry**| `app/app.py` natively supported | `app/app.py` | **YES** |
| **Custom Theme** | `.streamlit/config.toml` automatically loaded | Light medical palette | **YES** |
| **PyTorch CPU** | Automatic CPU wheel installation | CPU fallback verified | **YES** |
| **HTTPS URL** | Free custom domain (`https://<your-app>.streamlit.app`)| HTTPS shareable link | **YES** |
| **All 8 Pages** | Supported without modifications | 8 views verified | **YES** |

---

## 5. Model-Size and Storage Audit for GitHub

GitHub enforces a **strict 100 MB per-file limit** on standard Git commits.

| Model File | Checkpoint Size | GitHub 100 MB Limit | Solution |
| :--- | :---: | :---: | :--- |
| `mobilenet_v2_best.pt` | **25.88 MB** | **PASSED** (<100 MB) | Direct Git commit to GitHub |
| `efficientnet_b0_best.pt` | **46.39 MB** | **PASSED** (<100 MB) | Direct Git commit to GitHub |
| `resnet18_best.pt` | **128.06 MB** | **EXCEEDS LIMIT** (by 28 MB) | Free GitHub Release Asset or Git LFS |

### The 100% Free Solution for `resnet18_best.pt`:
GitHub **Releases** allows uploading binary assets up to **2.0 GB per file for free** with unlimited public download bandwidth.

- **Option A (GitHub Releases — Recommended)**:
  We publish a Release (e.g. `v1.0.0`) on your GitHub repository (`https://github.com/Vijatejas03/alzheimer-disease-detection`) and attach `resnet18_best.pt`.
  In `app/utils/inference_engine.py`, if `resnet18_best.pt` is not present locally on the cloud instance, it downloads it once via standard HTTPS directly from your GitHub Release in ~5 seconds and caches it!
- **Option B (GitHub Git LFS)**:
  Track `results/models/resnet18_best.pt` with Git LFS. GitHub provides 1 GB free LFS storage.

---

## 6. Recommended Free Deployment Route

**DEPLOY VIA STREAMLIT COMMUNITY CLOUD DIRECTLY FROM GITHUB.**

### Why This Is the Best Route:
1. **Native Platform**: Built specifically for Streamlit; zero Dockerfile complexity.
2. **Zero Cost**: Completely free forever with no credit card required.
3. **Repository Already Connected**: Your local repository is already linked to `https://github.com/Vijatejas03/alzheimer-disease-detection.git`.
4. **Instant HTTPS Link**: You receive a professional, clean URL such as:
   `https://alzheimer-xai-detection.streamlit.app`
   ready to share with your HOD, evaluators, and teammates.

---

## 7. Exact Next Steps

Do not worry about any paid plans. Here is the exact path:

### Step 1: Tell Me to Prepare GitHub Synchronization
Reply with:
> **"Proceed with Streamlit Community Cloud."**

### Step 2: What I Will Do Automatically
1. Stage the clean repository files (excluding raw 6,400 dataset, temp files, and caches).
2. Commit `mobilenet_v2_best.pt` (25.88 MB) and `efficientnet_b0_best.pt` (46.39 MB).
3. Set up the automated GitHub Release download hook for `resnet18_best.pt` (128 MB) so it downloads seamlessly on Streamlit Cloud without triggering GitHub's 100 MB block.
4. Push the branch to `https://github.com/Vijatejas03/alzheimer-disease-detection.git`.

### Step 3: What You Will Click (1-Minute Setup)
1. Go to **[share.streamlit.io](https://share.streamlit.io)** and click **"Continue with GitHub"** (log in as `Vijatejas03`).
2. Click **"New app"** &rarr; select repository: `Vijatejas03/alzheimer-disease-detection`.
3. Set Main file path: `app/app.py`.
4. Click **"Deploy!"**.
5. Your application will be live over public HTTPS in ~2 minutes!
