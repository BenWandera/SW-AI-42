# 🎓 Active Learning System - Quick Start

## ✨ What's New?

Your waste classification AI can now **learn from user feedback** and continuously improve its accuracy!

## 🚀 Quick Start (3 Steps)

### Step 1: Start the API
```bash
cd api
python real_api.py
```

You should see:
```
🚀 Starting Waste Management API (Real MobileViT)
✅ MobileViT model ready!
🎓 Initializing active learning system...
✅ Active learning ready! 0 feedback samples collected
📡 API ready!
```

### Step 2: Test Active Learning
```bash
python test_active_learning.py
```

Choose option **7** (Run Full Demo) to see the complete system in action!

### Step 3: Integrate in Your App
Add feedback functionality to your Flutter app (see examples below).

## 📱 Flutter Integration Example

### 1. Add Feedback Button After Classification

```dart
// After showing classification result
Widget _buildFeedbackSection(ClassificationResult result) {
  return Card(
    child: Column(
      children: [
        Text('Was this classification correct?'),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            ElevatedButton.icon(
              icon: Icon(Icons.check),
              label: Text('Correct'),
              onPressed: () => _submitFeedback(
                isCorrect: true,
                correctClass: result.categoryName,
              ),
            ),
            ElevatedButton.icon(
              icon: Icon(Icons.close),
              label: Text('Incorrect'),
              onPressed: () => _showCorrectionDialog(result),
            ),
          ],
        ),
      ],
    ),
  );
}
```

### 2. Submit Feedback Function

```dart
Future<void> _submitFeedback({
  required bool isCorrect,
  required String correctClass,
}) async {
  final request = http.MultipartRequest(
    'POST',
    Uri.parse('$apiBaseUrl/api/feedback/submit'),
  );
  
  // Add form fields
  request.fields['user_id'] = _userId;
  request.fields['predicted_class'] = _lastPrediction.categoryName;
  request.fields['predicted_confidence'] = _lastPrediction.confidence.toString();
  request.fields['correct_class'] = correctClass;
  request.fields['is_correct'] = isCorrect.toString();
  
  // Add image file
  request.files.add(
    await http.MultipartFile.fromPath('image', _imageFile.path),
  );
  
  // Send request
  final response = await request.send();
  
  if (response.statusCode == 200) {
    final responseData = await response.stream.bytesToString();
    final jsonData = json.decode(responseData);
    
    // Show thank you message
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('✅ ${jsonData['message']}'),
        backgroundColor: Colors.green,
      ),
    );
    
    // Award bonus points for feedback
    _awardFeedbackPoints(10);
  }
}
```

### 3. Correction Dialog

```dart
void _showCorrectionDialog(ClassificationResult result) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text('Select Correct Category'),
      content: DropdownButton<String>(
        value: _selectedCorrection,
        items: [
          'Cardboard', 'Food Organics', 'Glass', 'Metal',
          'Miscellaneous Trash', 'Paper', 'Plastic',
          'Textile Trash', 'Vegetation'
        ].map((category) => DropdownMenuItem(
          value: category,
          child: Text(category),
        )).toList(),
        onChanged: (value) {
          setState(() => _selectedCorrection = value);
        },
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: () {
            Navigator.pop(context);
            _submitFeedback(
              isCorrect: false,
              correctClass: _selectedCorrection!,
            );
          },
          child: Text('Submit'),
        ),
      ],
    ),
  );
}
```

## 📊 API Endpoints

### Submit Feedback
```http
POST /api/feedback/submit
Content-Type: multipart/form-data

user_id: string
image: file
predicted_class: string
predicted_confidence: float
correct_class: string
is_correct: boolean
```

### Get Statistics
```http
GET /api/feedback/statistics
```

### View Dashboard
```http
GET /api/learning/dashboard
```

### Trigger Retraining
```http
POST /api/model/retrain?epochs=3&batch_size=8
```

## 📈 Benefits

### For Users:
✅ More accurate predictions over time  
✅ Personalized to local waste types  
✅ Rewards for providing feedback  
✅ See their impact on AI improvement  

### For You:
✅ Continuous model improvement  
✅ No manual retraining needed  
✅ Adapts to new waste types automatically  
✅ Handles edge cases better  
✅ Detailed performance analytics  

## 🎯 How It Works

```
1. User uploads waste image
2. Model predicts category
3. User confirms or corrects prediction
4. Feedback stored with priority
5. System collects 100+ feedbacks
6. Automatic retraining triggered
7. Updated model deployed
8. Improved accuracy! 🎉
```

## 📂 Files Overview

| File | Purpose |
|------|---------|
| `api/active_learning_system.py` | Core active learning logic |
| `api/model_retrainer.py` | Model retraining engine |
| `api/real_api.py` | API with active learning endpoints |
| `test_active_learning.py` | Testing and demo script |
| `ACTIVE_LEARNING_GUIDE.md` | Detailed documentation |

## 🔍 Monitoring

Check the learning dashboard regularly:
```bash
curl http://localhost:8000/api/learning/dashboard | jq
```

Key metrics to watch:
- **Overall Accuracy**: Should increase over time
- **Samples Ready**: When ≥100, retraining is recommended
- **Class Accuracy**: Identify weak categories
- **Confusion Matrix**: See common mistakes

## ⚙️ Configuration

Edit thresholds in `api/active_learning_system.py`:

```python
ActiveLearningManager(
    retrain_threshold=100,      # Samples needed
    retrain_interval_days=7     # Days between retrains
)
```

## 🧪 Testing Checklist

- [ ] API starts successfully with active learning
- [ ] Can submit feedback via API
- [ ] Statistics endpoint works
- [ ] Dashboard shows correct data
- [ ] Can simulate feedback with test script
- [ ] Retraining completes successfully
- [ ] Model backups are created

## 💡 Best Practices

1. **Ask for feedback on uncertain predictions** (confidence < 85%)
2. **Reward users** for providing feedback (bonus points)
3. **Show impact**: Display accuracy improvements to users
4. **Monitor regularly**: Check dashboard weekly
5. **Retrain carefully**: Validate before deploying

## 🚨 Troubleshooting

### "Feedback storage not initialized"
→ Restart the API server

### "Not enough samples for retraining"
→ Collect more feedback (need ≥20 samples)

### "Retraining failed"
→ Check logs in `feedback_data/`  
→ Ensure model file exists  
→ Verify disk space for backups  

## 📚 Full Documentation

See **`ACTIVE_LEARNING_GUIDE.md`** for:
- Complete architecture details
- All API endpoint specifications
- Advanced configuration options
- Monitoring and maintenance guide

## 🎉 You're Ready!

Your AI now learns continuously from real users! 🚀

**Next Steps:**
1. ✅ Test the system with demo script
2. ✅ Integrate feedback UI in Flutter app
3. ✅ Monitor feedback collection
4. ✅ Review first retraining results
5. ✅ Track accuracy improvements

---

**Questions?** Check `ACTIVE_LEARNING_GUIDE.md` or review the code!

**Built with ❤️ for continuous improvement! 🎓🌍♻️**
