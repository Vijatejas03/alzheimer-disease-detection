# Git & GitHub Setup Guide

## Current Git Status

Portable Git 2.47.1 is installed at:
  C:\Users\vijay\AppData\Local\PortableGit\cmd\git.exe

The repository has been initialised and the first commit has been created.

---

## Using Git from the VS Code Terminal or PowerShell

Portable Git is NOT automatically on the system PATH.
To use git commands, either:

### Option A — Add to your PowerShell session temporarily
```powershell
$env:PATH = "C:\Users\vijay\AppData\Local\PortableGit\cmd;" + $env:PATH
git --version   # should now work
```

### Option B — Add permanently to your user PATH (recommended)
1. Open Start > search "Environment Variables"
2. Click "Edit the system environment variables" > "Environment Variables"
3. Under "User variables", select "Path" and click "Edit"
4. Click "New" and add:
   C:\Users\vijay\AppData\Local\PortableGit\cmd
5. Click OK. Restart VS Code or PowerShell.

### Option C — Use the full path in scripts
```powershell
$git = "C:\Users\vijay\AppData\Local\PortableGit\cmd\git.exe"
& $git status
& $git log --oneline
```

### Option D — Install Git for Windows properly (requires administrator)
Download from https://git-scm.com/download/win and run as Administrator.
This adds git to the PATH globally, so it works everywhere without any extra steps.

---

## Repository Status

- Repository location : C:\Users\vijay\.gemini\antigravity\scratch\Alzheimer_Disease_Detection
- Branch              : main
- First commit        : 2b5e3fd  "Initial Alzheimer disease detection project setup"
- Files committed     : 36

---

## GitHub — Next Steps

No remote repository has been configured yet.

### Step 1 — Create a GitHub Repository

1. Go to https://github.com/new
2. Set repository name (e.g., `alzheimer-disease-detection`)
3. Set to Public or Private (your choice)
4. **Do NOT** tick "Add a README file", "Add .gitignore", or "Choose a license"
   (we already have these — adding them on GitHub would cause a merge conflict)
5. Click **"Create repository"**
6. GitHub will show you a page with the remote URL. Copy it.

### Step 2 — Add the Remote

In your project terminal (with git on PATH):
```bash
git remote add origin https://github.com/<your-username>/alzheimer-disease-detection.git
```

### Step 3 — Push

```bash
git branch -M main
git push -u origin main
```

Git will open a browser window for GitHub authentication (via Git Credential Manager).
You do NOT need to paste a token or password manually.

---

## What Is and Is NOT Committed

### COMMITTED (source code only)
- All Python source files in src/
- inspect_dataset.py, experiment_config.json
- README.md, METHODOLOGY.md, GIT_SETUP.md
- requirements.txt, .gitignore, .gitattributes
- Directory placeholder .gitkeep files

### NOT COMMITTED (excluded by .gitignore)
- data/dataset/ images
- results/models/*.pt checkpoints
- results/figures/*.png generated plots
- results/metrics/*.json experiment outputs
- __pycache__/ Python bytecode
- Virtual environments (.venv/, venv/)
- Logs (*.log)

---

## Recommended Future Commit Messages

| Stage | Commit Message |
|:---|:---|
| Dataset confirmed | `Confirm dataset: N images, 4 classes, inspection passed` |
| Training complete | `Train MobileNetV2: best val acc saved to checkpoint` |
| Evaluation | `Evaluate all 3 models: metrics and confusion matrices` |
| Web app added | `Add Streamlit web application` |
| Final submission | `Final submission: all experiments complete` |
