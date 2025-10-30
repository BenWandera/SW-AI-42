# 📱 EcoWaste AI - Flutter Mobile App

AI-powered waste management and classification mobile application built with Flutter.

## 🎯 Overview

EcoWaste AI is a comprehensive waste management mobile application that uses state-of-the-art AI models (MobileViT + GNN) to classify waste items, provide disposal instructions, and gamify recycling through an incentive system.

### Key Features

- 📸 **Camera & Image Classification** - Capture or select images for instant waste classification
- 🤖 **Dual AI System** - MobileViT (88.42%) + GNN reasoning (93.26% accuracy)
- 💰 **Incentive System** - Earn points (10-264 per sort) with streak multipliers up to 1.5x
- 🏆 **Gamification** - Achievements, tiers (Bronze→Silver→Gold→Platinum), leaderboards
- 🎁 **Rewards Marketplace** - Redeem points for digital/physical rewards
- 📊 **Personal Analytics** - Track sorting history, accuracy, and environmental impact
- 🗺️ **Kampala Integration** - Division maps, KCCA data, recycling center locator
- 🌍 **9 Waste Categories** - Plastic, Paper, Organic, Vegetation, Glass, Metal, Electronic, Medical, Misc

## 🏗️ Project Structure

```
waste_flutter_app/
├── lib/
│   ├── main.dart                 # App entry point & navigation
│   ├── models/                   # Data models
│   │   ├── waste_category.dart   # 9 waste categories with properties
│   │   ├── classification_result.dart  # AI classification results
│   │   ├── user_model.dart       # User profile and stats
│   │   └── incentive_models.dart # Points, achievements, rewards
│   ├── services/                 # Business logic
│   │   └── api_service.dart      # Backend API integration
│   ├── screens/                  # UI screens
│   │   ├── home_screen.dart      # Main dashboard
│   │   ├── classification_screen.dart  # Classification loading
│   │   ├── result_screen.dart    # Classification results
│   │   ├── profile_screen.dart   # User profile
│   │   ├── rewards_screen.dart   # Rewards marketplace
│   │   ├── history_screen.dart   # Classification history
│   │   └── leaderboard_screen.dart  # Leaderboards
│   └── widgets/                  # Reusable components
│       ├── category_card.dart    # Category display card
│       ├── stats_card.dart       # Statistics card
│       └── loading_animation.dart # Custom loading animation
├── assets/                       # Images, icons, animations
├── pubspec.yaml                  # Dependencies
└── README.md                     # This file
```

## 📦 Dependencies

### Core Dependencies
- **flutter** - UI framework
- **google_fonts** - Typography (Poppins)
- **provider** / **get** - State management

### Camera & Image
- **camera** (^0.10.5) - Camera access
- **image_picker** (^1.0.4) - Gallery selection
- **image** (^4.1.3) - Image processing

### Networking
- **http** (^1.1.0) - HTTP client
- **dio** (^5.4.0) - Advanced networking

### Storage & Data
- **shared_preferences** (^2.2.2) - Local storage
- **sqflite** (^2.3.0) - Local database

### UI & Visualization
- **fl_chart** (^0.65.0) - Charts and graphs
- **percent_indicator** (^4.2.3) - Progress indicators
- **shimmer** (^3.0.0) - Loading effects
- **lottie** (^3.0.0) - Animations

### Location & Maps
- **geolocator** (^10.1.0) - Location services
- **google_maps_flutter** (^2.5.0) - Maps integration

### Utilities
- **logger** (^2.0.2) - Logging
- **qr_flutter** (^4.1.0) - QR code generation
- **cached_network_image** (^3.3.0) - Image caching

## 🚀 Getting Started

### Prerequisites

1. **Install Flutter** (3.0.0 or higher)
   ```bash
   # Check Flutter installation
   flutter --version
   ```

2. **Install Android Studio** or **Xcode** (for iOS)

3. **Install VS Code** (optional but recommended)
   - Flutter extension
   - Dart extension

### Installation

1. **Navigate to the Flutter app directory**
   ```bash
   cd waste_flutter_app
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Configure API endpoint**
   
   Edit `lib/services/api_service.dart`:
   ```dart
   static const String baseUrl = 'http://YOUR_BACKEND_URL:8000/api';
   ```

4. **Run the app**
   ```bash
   # List available devices
   flutter devices

   # Run on connected device
   flutter run

   # Run on specific device
   flutter run -d <device_id>

   # Run in release mode
   flutter run --release
   ```

### Backend Setup

The app requires a Python FastAPI backend with these endpoints:

```
POST   /api/classify              # Classify waste image
POST   /api/incentive/calculate   # Calculate incentive points
GET    /api/users/:id             # Get user profile
PUT    /api/users/:id             # Update user profile
GET    /api/users/:id/achievements  # Get achievements
GET    /api/rewards               # Get available rewards
POST   /api/rewards/redeem        # Redeem reward
GET    /api/leaderboard           # Get leaderboard
GET    /api/users/:id/statistics  # Get user statistics
GET    /api/users/:id/history     # Get classification history
POST   /api/auth/login            # User login
POST   /api/auth/register         # User registration
```

## 🎨 UI/UX Design

### Color Scheme
- **Primary**: #4CAF50 (Green) - Environmental theme
- **Secondary**: Various category colors (see waste_category.dart)
- **Accent**: #FFD700 (Gold) - Points and achievements

### Typography
- **Font Family**: Poppins (via Google Fonts)
- **Weights**: 400 (Regular), 500 (Medium), 600 (SemiBold), 700 (Bold)

### Design Principles
- Material Design 3 (Material You)
- Clean, modern interface
- Intuitive navigation with bottom nav bar
- Card-based layouts
- Smooth animations and transitions
- Dark mode support

## 📱 App Flow

```
1. Home Screen
   ├─ Welcome header with user info
   ├─ Stats cards (Points, Streak, Tier)
   ├─ Camera button (main action)
   ├─ Waste categories grid
   └─ Quick actions

2. Camera/Gallery Selection
   └─ Choose camera or gallery

3. Classification Screen
   ├─ Image preview
   ├─ Loading animation
   ├─ MobileViT classification (0-30%)
   ├─ GNN reasoning (30-60%)
   └─ Incentive calculation (60-100%)

4. Result Screen
   ├─ Classified category
   ├─ Confidence score
   ├─ AI analysis breakdown
   ├─ Points earned
   ├─ New achievements (if any)
   ├─ Disposal instructions
   └─ Action buttons

5. Other Screens
   ├─ Profile - User stats and settings
   ├─ Leaderboard - Rankings
   ├─ History - Past classifications
   └─ Rewards - Marketplace
```

## 🔧 Configuration

### Android Configuration

Edit `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA"/>
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
<uses-permission android:name="android.permission.INTERNET"/>
<uses-permission android:name="android.permission.ACCESS_FINE_LOCATION"/>
<uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION"/>
```

### iOS Configuration

Edit `ios/Runner/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>We need camera access to scan waste items</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need photo library access to select waste images</string>
<key>NSLocationWhenInUseUsageDescription</key>
<string>We need location to find nearby recycling centers</string>
```

## 🧪 Testing

```bash
# Run all tests
flutter test

# Run tests with coverage
flutter test --coverage

# Run integration tests
flutter drive --target=test_driver/app.dart
```

## 📦 Building for Production

### Android APK
```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

### Android App Bundle (Google Play)
```bash
flutter build appbundle --release
# Output: build/app/outputs/bundle/release/app-release.aab
```

### iOS (requires macOS)
```bash
flutter build ios --release
# Open Xcode and archive
```

## 🎯 Waste Categories

| Category | Icon | Color | Base Points | Recyclable | Hazardous |
|----------|------|-------|-------------|------------|-----------|
| Plastic | 🥤 | Blue | 15 | ✅ | ❌ |
| Paper & Cardboard | 📄 | Brown | 12 | ✅ | ❌ |
| Organic/Food | 🍽️ | Orange | 10 | ❌ | ❌ |
| Vegetation | 🌿 | Green | 8 | ❌ | ❌ |
| Glass | 🍷 | Cyan | 18 | ✅ | ❌ |
| Metal | ⚙️ | Gray | 20 | ✅ | ❌ |
| Electronic | 📱 | Purple | 30 | ✅ | ⚠️ |
| Medical | 💉 | Red | 50 | ❌ | ⚠️ |
| Miscellaneous | 🗑️ | Gray | 5 | ❌ | ❌ |

## 💡 AI Model Integration

### Classification Pipeline

1. **MobileViT Classification** (88.42% accuracy)
   - Initial waste categorization
   - Fast inference (<500ms)
   - 9 waste categories

2. **GNN Reasoning** (→93.26% accuracy)
   - Knowledge graph validation
   - Misclassification correction
   - Safety conflict detection
   - 21 nodes, 23 edges

3. **Incentive Calculation**
   - Base points per category
   - Confidence multiplier
   - Streak bonus (up to 1.5x)
   - Achievement unlocks

### Model Update Strategy

```dart
// In api_service.dart
static const String modelVersion = '1.0.0';

// TODO: Implement model versioning
// - Check for updates on app start
// - Download updated models
// - Switch to new model seamlessly
```

## 🔐 Security & Privacy

- ✅ Images processed temporarily (not stored on server)
- ✅ JWT authentication for API calls
- ✅ HTTPS encryption in transit
- ✅ Local data encrypted with sqflite_cipher
- ✅ User data deletion on request
- ✅ GDPR compliant

## 🐛 Troubleshooting

### Common Issues

**1. Camera not working**
```bash
# Check permissions in AndroidManifest.xml or Info.plist
# Restart the app
```

**2. API connection failed**
```dart
// Check API endpoint in lib/services/api_service.dart
// Ensure backend is running
// Check network connection
```

**3. Build failed**
```bash
# Clean and rebuild
flutter clean
flutter pub get
flutter run
```

**4. Dependencies conflict**
```bash
# Update dependencies
flutter pub upgrade
```

## 📊 Performance Optimization

- Image compression before upload (max 1024x1024, 85% quality)
- Cached network images
- Lazy loading for lists
- Efficient local caching with shared_preferences
- Background processing for heavy operations
- Optimized widgets with const constructors

## 🚀 Future Enhancements

- [ ] Offline mode with TFLite models
- [ ] AR waste identification
- [ ] Social features (sharing, challenges)
- [ ] Push notifications
- [ ] Multi-language support (English, Luganda, Swahili)
- [ ] Dark mode enhancements
- [ ] Accessibility improvements
- [ ] Payment integration for rewards
- [ ] Real-time leaderboards with WebSocket

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ben Wandera**
- GitHub: [@BenWandera](https://github.com/BenWandera)
- Project: [SW-AI-42](https://github.com/BenWandera/SW-AI-42)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues or questions:
- Create an issue on GitHub
- Email: your-email@example.com

---

**Built with ❤️ for a cleaner environment** 🌍♻️
