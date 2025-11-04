#!/bin/bash
# Setup Git LFS and Upload Everything to GitHub
# This will upload ALL files including large models

echo "🚀 Git LFS Setup - Upload Everything to GitHub"
echo "================================================================"

# Step 1: Check if Git LFS is installed
echo ""
echo "1️⃣ Checking Git LFS installation..."
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed!"
    echo ""
    echo "📥 Install Git LFS:"
    echo "   Windows: Download from https://git-lfs.github.com/"
    echo "   Or use: winget install -e --id GitHub.GitLFS"
    echo ""
    echo "After installation, run this script again."
    exit 1
else
    echo "   ✅ Git LFS is installed"
    git lfs version
fi

# Step 2: Initialize Git LFS
echo ""
echo "2️⃣ Initializing Git LFS in repository..."
git lfs install
echo "   ✅ Git LFS initialized"

# Step 3: Track large file types with Git LFS
echo ""
echo "3️⃣ Configuring Git LFS to track large files..."

# Track model files
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.ckpt"
git lfs track "*.bin"
git lfs track "*.h5"
git lfs track "*.pkl"

# Track large archives
git lfs track "*.zip"
git lfs track "*.tar.gz"
git lfs track "*.rar"

echo "   ✅ Git LFS tracking configured"
echo ""
echo "   Tracking:"
echo "     • *.pth (PyTorch models)"
echo "     • *.pt (PyTorch tensors)"
echo "     • *.ckpt (Checkpoints)"
echo "     • *.bin (Binary models)"
echo "     • *.h5 (Keras/HDF5 models)"
echo "     • *.zip, *.tar.gz (Archives)"

# Step 4: Update .gitignore to allow large files
echo ""
echo "4️⃣ Updating .gitignore to include everything..."

cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment (still excluded)
.venv/
venv/
ENV/
env/

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db
desktop.ini

# Temporary files
*.tmp
*.temp
.pytest_cache/
.coverage
htmlcov/

# Git LFS objects (don't commit these)
.git/lfs/

# Only exclude organize_files.py
organize_files.py
EOF

echo "   ✅ .gitignore updated (now allowing models, images, data)"

# Step 5: Add .gitattributes for LFS
echo ""
echo "5️⃣ Creating .gitattributes for Git LFS..."
git add .gitattributes
echo "   ✅ .gitattributes added"

# Step 6: Show what will be uploaded
echo ""
echo "6️⃣ Files to be uploaded:"
echo "   ═══════════════════════════════════════"

# Count files
TOTAL_FILES=$(git ls-files | wc -l)
LFS_FILES=$(git lfs ls-files 2>/dev/null | wc -l)

echo "   📦 Total files: $TOTAL_FILES"
echo "   🔷 LFS-tracked files: $LFS_FILES"
echo ""
echo "   Large files that will use Git LFS:"
find . -name "*.pth" -o -name "*.pt" -o -name "*.bin" | grep -v ".venv" | grep -v ".git" | head -10

# Step 7: Stage all changes
echo ""
echo "7️⃣ Staging all files..."
git add .

echo "   ✅ All files staged"

# Step 8: Show status
echo ""
echo "8️⃣ Git Status:"
git status --short | head -20
echo "   ... (showing first 20 files)"

# Step 9: Commit
echo ""
echo "9️⃣ Committing changes..."
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Add all files including models, datasets, and outputs using Git LFS"
fi

git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo "   ⚠️ Nothing new to commit or commit failed"
fi

# Step 10: Push everything
echo ""
echo "🔟 Pushing everything to GitHub..."
echo "   ⚠️ This may take a while for large files..."
echo ""

git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ SUCCESS! Everything uploaded to GitHub!"
    echo "================================================================"
    echo ""
    echo "📊 Upload Summary:"
    echo "   • All code files"
    echo "   • All model checkpoints (*.pth, *.pt)"
    echo "   • All images and visualizations"
    echo "   • All datasets (if included)"
    echo "   • All logs and outputs"
    echo ""
    echo "🔷 Large files are stored in Git LFS"
    echo "📏 Repository size on GitHub: ~1.5 GB"
    echo ""
    echo "🌐 View at: https://github.com/BenWandera/SW-AI-42"
    echo ""
    echo "💡 Note: Users will need Git LFS to clone:"
    echo "   git lfs install"
    echo "   git clone https://github.com/BenWandera/SW-AI-42.git"
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    echo "Common issues:"
    echo "   1. Git LFS quota exceeded (GitHub free: 1GB/month)"
    echo "   2. File too large (LFS limit: 2GB per file)"
    echo "   3. Network issues"
    echo ""
    echo "Solutions:"
    echo "   • Upgrade to GitHub Pro for more LFS bandwidth"
    echo "   • Split very large files"
    echo "   • Use alternative storage (HuggingFace Hub, S3)"
fi

echo ""
echo "================================================================"
