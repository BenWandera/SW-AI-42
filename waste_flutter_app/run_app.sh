#!/bin/bash

# Script to run Flutter app with correct Java version

echo "=========================================="
echo "Flutter Waste Management App Runner"
echo "=========================================="
echo ""

# Set JAVA_HOME to Android Studio's JBR (Java 17)
export JAVA_HOME="/c/Program Files/Android/Android Studio/jbr"
export PATH="$JAVA_HOME/bin:$PATH"

echo "Using Java version:"
java -version
echo ""

# Stop any running Gradle daemons
echo "Stopping Gradle daemons..."
cd android && ./gradlew --stop 2>/dev/null
cd ..

# Clean build
echo "Cleaning Flutter build..."
flutter clean

# Run the app
echo "Running Flutter app..."
flutter run

echo ""
echo "=========================================="
