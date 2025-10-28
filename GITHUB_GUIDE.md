# GitHub Push Quick Guide 🚀

## Prerequisites
- ✅ Git installed ([Download here](https://git-scm.com/))
- ✅ GitHub account created
- ✅ All files ready to commit

## Method 1: Use the Setup Script (Easiest)

### Windows:
```bash
github_setup.bat
```

### Linux/Mac:
```bash
bash github_setup.sh
```

Then follow the on-screen instructions!

## Method 2: Manual Setup

### Step 1: Initialize Git Repository
```bash
git init
```

### Step 2: Add Files
```bash
git add .gitignore .gitattributes LICENSE README_GITHUB.md
git add *.py *.json *.md requirements.txt
git add "GNN model/"
```

### Step 3: Commit Changes
```bash
git commit -m "Initial commit: AI Waste Management System"
```

### Step 4: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `waste-management-ai`
3. Description: `AI-powered waste classification with Vision Transformers and GNN`
4. Choose Public or Private
5. **DO NOT** initialize with README (we have one)
6. Click "Create repository"

### Step 5: Connect to GitHub
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/waste-management-ai.git
```

### Step 6: Push to GitHub
```bash
git branch -M main
git push -u origin main
```

## Handling Large Files (Model Checkpoints)

### Option 1: Use Git LFS (Recommended for models)
```bash
# Install Git LFS from https://git-lfs.github.com/
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add best_mobilevit_waste_model.pth
git commit -m "Add model checkpoint with LFS"
git push
```

### Option 2: Don't Commit Models (Already in .gitignore)
The `.gitignore` file already excludes `*.pth` files by default.
Instead, provide download links in your README:
```markdown
## Model Checkpoints
Download pre-trained models:
- [MobileViT](https://drive.google.com/...)
- [DeiT-Tiny](https://drive.google.com/...)
```

## Common Issues & Solutions

### Issue: "fatal: not a git repository"
**Solution:**
```bash
git init
```

### Issue: "remote origin already exists"
**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/waste-management-ai.git
```

### Issue: File too large (>100MB)
**Solutions:**
1. Use Git LFS (see above)
2. Remove the file: `git rm --cached large_file.pth`
3. Add to `.gitignore` and commit

### Issue: Authentication failed
**Solutions:**
1. Use Personal Access Token instead of password
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope
   - Use token as password when pushing

2. Or use SSH:
```bash
git remote set-url origin git@github.com:YOUR_USERNAME/waste-management-ai.git
```

## What's Being Committed?

### ✅ Included (Safe to commit):
- Python scripts (`.py`)
- Configuration files (`.json`, `requirements.txt`)
- Documentation (`.md`)
- Small result files (JSON metrics)
- Git configuration (`.gitignore`, `.gitattributes`)

### ❌ Excluded (In .gitignore):
- Virtual environment (`.venv/`)
- Model checkpoints (`*.pth`) - Too large
- Dataset images (`realwaste/`, `*.png`, `*.jpg`)
- Cache files (`__pycache__/`, `*.pyc`)
- Logs (`*.log`, `*.txt`)
- Generated outputs (large images, temporary files)

## Updating Your Repository

After making changes:
```bash
# Check what changed
git status

# Add specific files
git add file1.py file2.py

# Or add all changed files
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## Repository Best Practices

### 1. Write Good Commit Messages
```bash
# Good ✅
git commit -m "Add DeiT-Tiny model comparison feature"
git commit -m "Fix GNN import path resolution"
git commit -m "Update README with installation instructions"

# Bad ❌
git commit -m "update"
git commit -m "fix stuff"
git commit -m "changes"
```

### 2. Commit Often
- Commit after completing a feature
- Commit after fixing a bug
- Commit before making major changes

### 3. Use Branches for Features
```bash
# Create feature branch
git checkout -b feature/new-classifier

# Work on feature...
git add .
git commit -m "Add new classifier"

# Merge back to main
git checkout main
git merge feature/new-classifier
```

### 4. Keep Repository Clean
- Don't commit large files without LFS
- Don't commit sensitive data (API keys, passwords)
- Don't commit generated files that can be recreated
- Use `.gitignore` properly

## Making Your Repo Look Professional

### Add Badges to README
```markdown
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
```

### Add Topics/Tags
On GitHub, go to your repo → About section → Add topics:
- `artificial-intelligence`
- `computer-vision`
- `waste-management`
- `pytorch`
- `transformers`
- `graph-neural-networks`

### Enable GitHub Pages (Optional)
1. Go to Settings → Pages
2. Choose main branch
3. Your documentation will be available at:
   `https://YOUR_USERNAME.github.io/waste-management-ai/`

## Quick Commands Reference

```bash
# Status
git status                    # Check current status
git log                       # View commit history
git diff                      # See changes

# Add & Commit
git add .                     # Add all files
git add file.py               # Add specific file
git commit -m "message"       # Commit with message

# Push & Pull
git push                      # Push changes to GitHub
git pull                      # Pull changes from GitHub

# Branches
git branch                    # List branches
git branch new-branch         # Create branch
git checkout branch-name      # Switch branch
git merge branch-name         # Merge branch

# Remote
git remote -v                 # View remotes
git remote add origin URL     # Add remote
git remote remove origin      # Remove remote
```

## Need Help?

- GitHub Docs: https://docs.github.com/
- Git Cheat Sheet: https://education.github.com/git-cheat-sheet-education.pdf
- Git LFS: https://git-lfs.github.com/

---

**Ready to push? Run the setup script or follow the manual steps above! 🚀**
