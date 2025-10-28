#!/bin/bash
# GitHub Repository Setup Script

echo "🚀 GitHub Repository Setup for Waste Management AI"
echo "=================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

echo "✓ Git is installed"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📂 Initializing Git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Create .gitattributes for Git LFS (for large files)
echo "📝 Creating .gitattributes for large files..."
cat > .gitattributes << 'EOF'
# Git LFS configuration for large files
*.pth filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
EOF

echo "✓ .gitattributes created"

# Add all files
echo "➕ Adding files to git..."
git add .gitignore
git add .gitattributes
git add LICENSE
git add README_GITHUB.md
git add requirements.txt
git add *.py
git add "GNN model/"
git add *.json
git add *.md

echo "✓ Files staged for commit"

# Show status
echo ""
echo "📊 Git Status:"
git status

echo ""
echo "=================================================="
echo "📝 Next Steps:"
echo "=================================================="
echo ""
echo "1. Review the staged files above"
echo ""
echo "2. Commit your changes:"
echo "   git commit -m 'Initial commit: AI Waste Management System'"
echo ""
echo "3. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name: waste-management-ai"
echo "   - Description: AI-powered waste classification with Vision Transformers and GNN"
echo "   - Make it Public or Private"
echo "   - Do NOT initialize with README (we already have one)"
echo ""
echo "4. Add GitHub as remote:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/waste-management-ai.git"
echo ""
echo "5. Rename branch to main (if needed):"
echo "   git branch -M main"
echo ""
echo "6. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "=================================================="
echo "📦 Optional: Set up Git LFS for large model files"
echo "=================================================="
echo ""
echo "If you want to include model files (.pth), install Git LFS:"
echo "   1. Download from: https://git-lfs.github.com/"
echo "   2. Install and run: git lfs install"
echo "   3. Track model files: git lfs track '*.pth'"
echo "   4. Commit and push as normal"
echo ""
echo "⚠️  Note: GitHub has a 100MB file size limit without LFS"
echo ""
echo "=================================================="
echo "✅ Setup complete! Follow the steps above to push to GitHub."
echo "=================================================="
