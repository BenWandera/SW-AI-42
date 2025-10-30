# Download SF Pro Font Files

## Instructions to get San Francisco (SF Pro) fonts:

### Option 1: Download from Apple (Official - Recommended)
1. Visit: https://developer.apple.com/fonts/
2. Click "Download" on the SF Pro font
3. Extract the downloaded .dmg or .zip file
4. Copy these files to `assets/fonts/`:
   - SF-Pro-Display-Regular.otf
   - SF-Pro-Display-Medium.otf
   - SF-Pro-Display-Semibold.otf
   - SF-Pro-Display-Bold.otf

### Option 2: Use Alternative Source
1. Visit: https://github.com/sahibjotsaggu/San-Francisco-Pro-Fonts
2. Download the repository
3. Copy the .otf files mentioned above to `assets/fonts/`

### Option 3: Use a Similar Font (Temporary)
If you can't access SF Pro immediately, you can use **Inter** font which is very similar:

1. Visit: https://fonts.google.com/specimen/Inter
2. Download the font family
3. Rename the files to match:
   - Inter-Regular.ttf → SF-Pro-Display-Regular.otf
   - Inter-Medium.ttf → SF-Pro-Display-Medium.otf
   - Inter-SemiBold.ttf → SF-Pro-Display-Semibold.otf
   - Inter-Bold.ttf → SF-Pro-Display-Bold.otf

## Files Needed in `assets/fonts/`:
```
waste_flutter_app/
  assets/
    fonts/
      SF-Pro-Display-Regular.otf    (400 weight)
      SF-Pro-Display-Medium.otf     (500 weight)
      SF-Pro-Display-Semibold.otf   (600 weight)
      SF-Pro-Display-Bold.otf       (700 weight)
```

## After Adding Fonts:
1. Run: `flutter clean`
2. Run: `flutter pub get`
3. Run: `flutter build apk --release`

The app will now use SF Pro font throughout!
