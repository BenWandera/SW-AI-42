# Active Learning Implementation Guide

## 🎓 Overview

Your waste classification system now includes **Active Learning** - a continuous improvement system that learns from user feedback to improve classification accuracy over time.

## 🌟 Key Features

### 1. **Feedback Collection**
- Users can correct wrong predictions
- System tracks all user feedback
- Stores images and corrections for retraining
- Calculates prediction confidence and priority

### 2. **Intelligent Sampling**
- Prioritizes uncertain predictions (low confidence)
- Prioritizes incorrect predictions
- Uses feedback to identify weak areas

### 3. **Automated Retraining**
- Automatically suggests retraining when:
  - ≥100 new feedback samples collected
  - 7+ days since last retrain (with ≥20 samples)
  - Model accuracy drops below 75%
  
### 4. **Model Versioning**
- Automatic backup before retraining
- Ability to restore previous versions
- Training history tracking

## 📊 How It Works

```
┌─────────────────┐
│  User Upload    │
│     Image       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MobileViT     │
│  Classifies     │──────┐
└────────┬────────┘      │
         │               │ Low Confidence?
         ▼               │ Ask for feedback
┌─────────────────┐      │
│  Show Result    │◄─────┘
│  to User        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ User Confirms   │
│  or Corrects    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store Feedback  │
│  + Statistics   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Enough samples? │──Yes──┐
└────────┬────────┘        │
         No                │
         │                 ▼
         │        ┌─────────────────┐
         │        │  Retrain Model  │
         │        │  (Incremental)  │
         │        └────────┬────────┘
         │                 │
         │                 ▼
         │        ┌─────────────────┐
         │        │  Deploy Updated │
         │        │     Model       │
         │        └─────────────────┘
         │
         └──────► Continue...
```

## 🔌 API Endpoints

### Submit Feedback
```http
POST /api/feedback/submit
Content-Type: multipart/form-data

Parameters:
- user_id: string
- image: file
- predicted_class: string
- predicted_confidence: float
- correct_class: string
- is_correct: boolean

Response:
{
  "success": true,
  "feedback_id": "feedback_123",
  "message": "Thank you for your feedback!",
  "statistics": {
    "total_feedback": 150,
    "accuracy": 87.5,
    "samples_ready": 45
  },
  "retraining": {
    "recommended": false,
    "reason": "Not ready for retraining"
  }
}
```

### Get Statistics
```http
GET /api/feedback/statistics

Response:
{
  "success": true,
  "statistics": {
    "total_feedback": 150,
    "correct_predictions": 132,
    "incorrect_predictions": 18,
    "overall_accuracy": 88.0,
    "class_accuracy": {
      "Plastic": {"accuracy": 92.5, "correct": 37, "total": 40},
      "Paper": {"accuracy": 85.0, "correct": 17, "total": 20}
    },
    "confusion_matrix": {
      "Plastic->Paper": 3,
      "Paper->Cardboard": 5
    },
    "avg_confidence": 0.89,
    "samples_ready_for_training": 45
  }
}
```

### Get Learning Dashboard
```http
GET /api/learning/dashboard

Response:
{
  "success": true,
  "dashboard": {
    "feedback_statistics": {...},
    "training_metadata": {
      "last_retrain": "2025-11-01T10:30:00",
      "retrain_count": 3,
      "total_samples_used": 285,
      "performance_history": [...]
    },
    "retraining": {
      "should_retrain": true,
      "reason": "Reached threshold: 105 new samples",
      "threshold": 100,
      "interval_days": 7
    },
    "recommendations": [
      "✅ Ready to retrain! 105 new samples available.",
      "⚠️ Plastic: Low accuracy (75.2%) - needs more training data"
    ]
  }
}
```

### Trigger Model Retraining
```http
POST /api/model/retrain?force=false&epochs=3&batch_size=8

Response:
{
  "success": true,
  "message": "Model retrained successfully!",
  "results": {
    "backup_path": "model_backups/model_backup_20251104_120000.pth",
    "epochs": 3,
    "training_samples": 84,
    "validation_samples": 21,
    "final_train_acc": 94.2,
    "final_val_acc": 91.5,
    "best_val_acc": 92.1,
    "training_history": [...]
  }
}
```

### List Model Backups
```http
GET /api/model/backups

Response:
{
  "success": true,
  "backups": [
    {
      "path": "model_backups/model_backup_20251104_120000.pth",
      "filename": "model_backup_20251104_120000.pth",
      "size_mb": 24.5,
      "created": "2025-11-04T12:00:00",
      "metadata": {...}
    }
  ],
  "total_backups": 3
}
```

### Restore Model Backup
```http
POST /api/model/restore
Content-Type: application/json

{
  "backup_filename": "model_backup_20251104_120000.pth"
}

Response:
{
  "success": true,
  "message": "Model restored from model_backup_20251104_120000.pth",
  "backup_path": "model_backups/model_backup_20251104_120000.pth"
}
```

## 🎯 Implementation in Flutter App

### 1. Add Feedback Button to Classification Result

```dart
// After showing classification result
if (confidence < 0.85) {
  // Show feedback prompt for uncertain predictions
  showDialog(
    context: context,
    builder: (context) => FeedbackDialog(
      predictedClass: result.categoryName,
      predictedConfidence: result.confidence,
      onFeedback: (correctClass, isCorrect) async {
        await submitFeedback(
          imageFile: imageFile,
          predictedClass: result.categoryName,
          predictedConfidence: result.confidence,
          correctClass: correctClass,
          isCorrect: isCorrect,
        );
      },
    ),
  );
}
```

### 2. Submit Feedback Function

```dart
Future<void> submitFeedback({
  required File imageFile,
  required String predictedClass,
  required double predictedConfidence,
  required String correctClass,
  required bool isCorrect,
}) async {
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('$baseUrl/api/feedback/submit'),
  );
  
  request.fields['user_id'] = userId;
  request.fields['predicted_class'] = predictedClass;
  request.fields['predicted_confidence'] = predictedConfidence.toString();
  request.fields['correct_class'] = correctClass;
  request.fields['is_correct'] = isCorrect.toString();
  
  request.files.add(
    await http.MultipartFile.fromPath('image', imageFile.path),
  );
  
  var response = await request.send();
  
  if (response.statusCode == 200) {
    var responseData = await response.stream.bytesToString();
    var jsonData = json.decode(responseData);
    
    // Show thank you message
    showSnackBar('Thank you for your feedback! ${jsonData['message']}');
    
    // Award points for providing feedback
    if (jsonData['statistics']['total_feedback'] % 10 == 0) {
      showSnackBar('Milestone! You\'ve provided ${jsonData['statistics']['total_feedback']} feedbacks!');
    }
  }
}
```

### 3. Feedback Dialog Widget

```dart
class FeedbackDialog extends StatefulWidget {
  final String predictedClass;
  final double predictedConfidence;
  final Function(String correctClass, bool isCorrect) onFeedback;
  
  @override
  _FeedbackDialogState createState() => _FeedbackDialogState();
}

class _FeedbackDialogState extends State<FeedbackDialog> {
  String? selectedClass;
  
  final List<String> classes = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal',
    'Miscellaneous Trash', 'Paper', 'Plastic', 
    'Textile Trash', 'Vegetation'
  ];
  
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: Text('Help Improve Our AI'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text('We predicted: ${widget.predictedClass}'),
          Text('Confidence: ${(widget.predictedConfidence * 100).toStringAsFixed(1)}%'),
          SizedBox(height: 16),
          Text('Is this correct?'),
          SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: () {
                  widget.onFeedback(widget.predictedClass, true);
                  Navigator.pop(context);
                },
                child: Text('✓ Correct'),
              ),
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    selectedClass = null;
                  });
                },
                child: Text('✗ Incorrect'),
              ),
            ],
          ),
          if (selectedClass == null && widget.predictedClass != selectedClass)
            DropdownButton<String>(
              hint: Text('Select correct class'),
              value: selectedClass,
              items: classes.map((c) => DropdownMenuItem(
                value: c,
                child: Text(c),
              )).toList(),
              onChanged: (value) {
                setState(() {
                  selectedClass = value;
                });
              },
            ),
          if (selectedClass != null)
            ElevatedButton(
              onPressed: () {
                widget.onFeedback(selectedClass!, false);
                Navigator.pop(context);
              },
              child: Text('Submit Correction'),
            ),
        ],
      ),
    );
  }
}
```

## 📈 Benefits

### For Users:
- ✅ More accurate classifications over time
- ✅ Personalized to common local waste types
- ✅ Rewards for providing feedback
- ✅ Transparency in AI improvement

### For System:
- ✅ Continuous accuracy improvement
- ✅ Automatic adaptation to new waste types
- ✅ Reduced manual data labeling
- ✅ Better performance on edge cases

## 🔐 Data Storage

### Feedback Data Structure:
```
feedback_data/
├── user_feedback.json          # All feedback records
├── training_metadata.json      # Retraining history
└── images/                     # User-corrected images
    ├── feedback_0_20251104120000.jpg
    ├── feedback_1_20251104120100.jpg
    └── ...
```

### Model Backups:
```
model_backups/
├── model_backup_20251104_120000.pth
├── model_backup_20251104_120000.json
├── model_backup_20251103_140000.pth
└── model_backup_20251103_140000.json
```

## ⚙️ Configuration

### Retraining Thresholds (in `active_learning_system.py`):
```python
ActiveLearningManager(
    feedback_storage=feedback_storage,
    retrain_threshold=100,      # Retrain after 100 new samples
    retrain_interval_days=7     # Or every 7 days (with ≥20 samples)
)
```

### Training Parameters (when calling retrain):
```python
model_retrainer.retrain(
    feedback_samples=samples,
    class_names=CLASS_NAMES,
    epochs=3,                   # Number of training epochs
    batch_size=8,               # Batch size
    validation_split=0.2        # 20% for validation
)
```

## 🧪 Testing

### 1. Submit Sample Feedback:
```bash
curl -X POST "http://localhost:8000/api/feedback/submit" \
  -F "user_id=test_user" \
  -F "image=@sample_waste.jpg" \
  -F "predicted_class=Plastic" \
  -F "predicted_confidence=0.75" \
  -F "correct_class=Metal" \
  -F "is_correct=false"
```

### 2. Check Statistics:
```bash
curl "http://localhost:8000/api/feedback/statistics"
```

### 3. View Dashboard:
```bash
curl "http://localhost:8000/api/learning/dashboard"
```

### 4. Trigger Retraining (after collecting enough feedback):
```bash
curl -X POST "http://localhost:8000/api/model/retrain?epochs=3&batch_size=8"
```

## 📊 Monitoring

### Key Metrics to Track:
1. **Overall Accuracy**: % of correct predictions
2. **Class-Specific Accuracy**: Accuracy per waste type
3. **Confusion Matrix**: Common misclassifications
4. **Average Confidence**: Model certainty
5. **Samples Ready**: Feedback ready for training
6. **Retraining Frequency**: How often model updates

### Dashboard Recommendations:
- Monitor for classes with <70% accuracy
- Look for repeated confusion patterns
- Track accuracy trends over time
- Review model performance after each retrain

## 🎓 Best Practices

1. **Collect Diverse Feedback**:
   - Encourage feedback on uncertain predictions
   - Balance feedback across all classes
   - Include edge cases and difficult examples

2. **Regular Retraining**:
   - Don't wait too long between retrains
   - But don't retrain with too few samples
   - Monitor performance after each retrain

3. **Quality Control**:
   - Review confused classes regularly
   - Validate model improvements
   - Keep backups of well-performing models

4. **User Engagement**:
   - Reward users for feedback
   - Show impact of their contributions
   - Make feedback process easy and quick

## 🚀 Future Enhancements

1. **Semi-Supervised Learning**: Use unlabeled data
2. **Hard Example Mining**: Actively seek difficult cases
3. **Multi-Model Ensemble**: Combine multiple models
4. **Transfer Learning**: Adapt from related domains
5. **Federated Learning**: Learn from distributed devices

## 📞 Support

For questions or issues with active learning:
- Check the logs in `feedback_data/`
- Review model backups in `model_backups/`
- Monitor API responses for error messages
- Contact the development team

---

**Built with ❤️ for continuous improvement and a cleaner planet! 🌍♻️**
