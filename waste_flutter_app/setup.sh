#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "   EcoWaste AI - Flutter App Setup"
echo "============================================"
echo ""

# Check if Flutter is installed
echo "[1/5] Checking Flutter installation..."
if ! command -v flutter &> /dev/null; then
    echo -e "${RED}[ERROR] Flutter is not installed!${NC}"
    echo ""
    echo "Please install Flutter from: https://flutter.dev/docs/get-started/install"
    echo "After installation, add Flutter to your PATH and restart this script."
    exit 1
fi
echo -e "${GREEN}[OK] Flutter is installed${NC}"
echo ""

# Check Flutter doctor
echo "[2/5] Running Flutter doctor..."
flutter doctor
echo ""

# Install dependencies
echo "[3/5] Installing Flutter dependencies..."
if ! flutter pub get; then
    echo -e "${RED}[ERROR] Failed to install dependencies${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Dependencies installed${NC}"
echo ""

# Check for connected devices
echo "[4/5] Checking for connected devices..."
flutter devices
echo ""

# Setup complete
echo "[5/5] Setup complete!"
echo ""
echo "============================================"
echo "   Setup Complete!"
echo "============================================"
echo ""
echo "Your Flutter app is ready to run."
echo ""
echo "To run the app:"
echo "  1. Connect your phone via USB (with USB debugging enabled)"
echo "  2. Or start an Android/iOS emulator"
echo "  3. Run: flutter run"
echo ""
echo "To build APK:"
echo "  - Debug:   flutter build apk --debug"
echo "  - Release: flutter build apk --release"
echo ""
echo "Documentation:"
echo "  - README.md       - Complete documentation"
echo "  - QUICKSTART.md   - Quick setup guide"
echo "  - PROJECT_SUMMARY.md - Feature overview"
echo ""
echo "============================================"
echo ""

# Ask if user wants to run the app
read -p "Do you want to run the app now? (y/n): " run
if [ "$run" = "y" ] || [ "$run" = "Y" ]; then
    echo ""
    echo "Starting app..."
    flutter run
else
    echo ""
    echo "You can run the app later with: flutter run"
fi

echo ""
echo "Press any key to exit..."
read -n 1 -s
