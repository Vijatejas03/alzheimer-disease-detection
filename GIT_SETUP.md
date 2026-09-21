# Git & GitHub Setup Guide

## Current Git Status

Git is **not installed** on this machine.

## Step 1 — Install Git for Windows

Download and install Git for Windows from the official source:

> https://git-scm.com/download/win

**Recommended installation options:**
- Use Git from the Windows Command Prompt
- Use the bundled OpenSSH
- Default branch name: `main`

After installation, restart your terminal or VS Code.

---

## Step 2 — Verify Installation

```bash
git --version
```
Expected output: `git version 2.x.x.windows.x`

---

## Step 3 — Configure Git Identity

```bash
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

---

## Step 4 — Initialise the Repository

From inside the project directory:

```bash
cd C:\Users\vijay\.gemini\antigravity\scratch\Alzheimer_Disease_Detection
git init
git add .
git commit -m "Initial commit: project structure, preprocessing, models, evaluation, inspection utility"
```

---

## Step 5 — Create GitHub Repository

1. Go to https://github.com/new
2. Create a **new empty repository** (do NOT initialise with README/gitignore — we already have these).
3. Copy the repository HTTPS or SSH URL.

---

## Step 6 — Add Remote and Push

```bash
git remote add origin https://github.com/<your-username>/<repo-name>.git
git branch -M main
git push -u origin main
```

> **IMPORTANT:** Do NOT push yet if you have not authenticated with GitHub.
> Set up a personal access token (PAT) or SSH key before pushing.

---

## What Will NOT Be Pushed (enforced by .gitignore)

- Dataset images (`data/dataset/**/*.jpg`, etc.)
- Model checkpoints (`results/models/*.pt`)
- Generated figures and metric JSONs
- Virtual environment folders
- Cache and log files

---

## Recommended Commit Structure

| Stage | Commit Message Example |
|:---|:---|
| Foundation | `Initial commit: project structure, src modules, inspection utility` |
| Dataset confirmed | `Confirm dataset structure: 4 classes, N images` |
| Preprocessing | `Add/update MRI preprocessing and augmentation pipeline` |
| Training complete | `Train MobileNetV2: epoch logs saved, checkpoint stored` |
| Evaluation | `Evaluate all 3 models: metrics and confusion matrices saved` |
| Web app | `Add Streamlit web application` |
| Final | `Final project submission: all experiments complete` |
