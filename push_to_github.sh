#!/bin/bash
# GitHub Push Script - Push to Main Branch
# Run this script to commit and push all changes to GitHub

echo "🚀 GitHub Push to Main Branch"
echo "======================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not initialized!"
    echo "Please run: git init"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Current branch: $CURRENT_BRANCH"

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "📝 Creating .gitignore..."
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

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebook
.ipynb_checkpoints/

# Model files (large files)
*.pth
*.pt
*.h5
*.pkl
*.ckpt

# Data directories
realwaste/
hf_cache/

# Output directories
mobilevit_metrics/
system_metrics/
gnn_visualizations/
gnn_correction_metrics/
waste_gan_output/
synthetic_outputs/
realwaste_eda_results/

# Logs
*.log
deit_training_log.txt

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
*.swp
EOF
    echo "   ✓ .gitignore created"
fi

# Stage all changes
echo ""
echo "📦 Staging changes..."
git add .

# Show status
echo ""
echo "📊 Git Status:"
git status --short

# Get commit message
echo ""
echo "💬 Enter commit message (or press Enter for default):"
read -r COMMIT_MSG

if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update: Complete waste management system with MobileViT, DeiT-Tiny, GNN, and GAN integration"
fi

# Commit changes
echo ""
echo "💾 Committing changes..."
git commit -m "$COMMIT_MSG"

if [ $? -ne 0 ]; then
    echo "⚠️  Nothing to commit or commit failed"
    echo "Continuing to push existing commits..."
fi

# Check if remote exists
REMOTE_URL=$(git remote get-url origin 2>/dev/null)

if [ -z "$REMOTE_URL" ]; then
    echo ""
    echo "🔗 No remote repository found!"
    echo "Enter your GitHub repository URL (e.g., https://github.com/username/repo.git):"
    read -r REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        echo "❌ No repository URL provided. Exiting."
        exit 1
    fi
    
    git remote add origin "$REPO_URL"
    echo "   ✓ Remote 'origin' added: $REPO_URL"
else
    echo "🔗 Remote repository: $REMOTE_URL"
fi

# Switch to main branch if needed
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo ""
    echo "🔄 Switching to main branch..."
    
    # Check if main branch exists
    if git show-ref --verify --quiet refs/heads/main; then
        git checkout main
    else
        # Create main branch from current branch
        git checkout -b main
    fi
    
    echo "   ✓ Now on main branch"
fi

# Push to main
echo ""
echo "⬆️  Pushing to GitHub main branch..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ Successfully pushed to GitHub!"
    echo "======================================"
    echo ""
    echo "🌐 Repository: $REMOTE_URL"
    echo "🌿 Branch: main"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Visit your GitHub repository"
    echo "   2. Add a description and topics"
    echo "   3. Enable GitHub Pages if needed"
    echo "   4. Share your project!"
else
    echo ""
    echo "❌ Push failed!"
    echo ""
    echo "Common solutions:"
    echo "   1. Make sure you're authenticated (git config --global credential.helper)"
    echo "   2. Check if you have push access to the repository"
    echo "   3. Try: git pull origin main --rebase"
    echo "   4. Then run this script again"
fi
