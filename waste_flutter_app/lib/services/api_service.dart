import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';
import '../models/classification_result.dart';
import '../models/incentive_models.dart';
import '../models/user_model.dart';

/// Main API service for communicating with the backend
class ApiService {
  // 🌐 PRODUCTION CONFIGURATION
  // Deployed on Render.com
  static const String PRODUCTION_URL = 'https://sw-ai-42.onrender.com'; // ✅ Live on Render!
  
  // 💻 LOCAL DEVELOPMENT CONFIGURATION
  static const String LOCAL_IP = '192.168.100.152'; // Your computer's IP for testing
  
  // 🔄 Switch between production and local
  // Set to true when releasing APK to users
  // Set to false when testing locally
  static const bool USE_PRODUCTION = true; // ✅ Using production server!
  
  static String get baseUrl {
    // Use production URL if enabled
    if (USE_PRODUCTION) {
      return '$PRODUCTION_URL/api';
    }
    
    // Otherwise use local development URL
    if (Platform.isAndroid) {
      // Check if running on emulator or real device
      // Real devices will use computer IP
      return 'http://$LOCAL_IP:8000/api';
    } else {
      return 'http://localhost:8000/api';
    }
  }
  
  final Logger _logger = Logger();
  final http.Client _client = http.Client();

  /// Classify an image using MobileViT + GNN
  Future<ClassificationResult> classifyImage(File imageFile) async {
    try {
      _logger.i('Classifying image: ${imageFile.path}');

      // Create multipart request
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/classify'),
      );

      // Add image file
      request.files.add(
        await http.MultipartFile.fromPath('image', imageFile.path),
      );

      // Send request
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        _logger.i('Classification successful: ${jsonData['category_name']}');
        return ClassificationResult.fromJson(jsonData);
      } else {
        _logger.e('Classification failed: ${response.statusCode}');
        throw Exception('Failed to classify image: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error classifying image: $e');
      rethrow;
    }
  }

  /// Calculate incentive points for a classification
  Future<IncentiveResult> calculateIncentive({
    required String userId,
    required String categoryId,
    required double confidence,
    required bool isCorrected,
  }) async {
    try {
      _logger.i('Calculating incentive for user: $userId');

      final response = await _client.post(
        Uri.parse('$baseUrl/incentive/calculate'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'user_id': userId,
          'category_id': categoryId,
          'confidence': confidence,
          'is_corrected': isCorrected,
        }),
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        _logger.i('Incentive calculated: ${jsonData['total_points']} points');
        return IncentiveResult.fromJson(jsonData);
      } else {
        throw Exception('Failed to calculate incentive: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error calculating incentive: $e');
      rethrow;
    }
  }

  /// Get user profile
  Future<UserModel> getUserProfile(String userId) async {
    try {
      _logger.i('Fetching user profile: $userId');

      final response = await _client.get(
        Uri.parse('$baseUrl/users/$userId'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        return UserModel.fromJson(jsonData);
      } else {
        throw Exception('Failed to get user profile: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting user profile: $e');
      rethrow;
    }
  }

  /// Update user profile
  Future<UserModel> updateUserProfile({
    required String userId,
    String? name,
    String? email,
    String? neighborhood,
    String? division,
  }) async {
    try {
      _logger.i('Updating profile for user: $userId');

      final queryParams = <String, String>{};
      if (name != null) queryParams['name'] = name;
      if (email != null) queryParams['email'] = email;
      if (neighborhood != null) queryParams['neighborhood'] = neighborhood;
      if (division != null) queryParams['division'] = division;

      final uri = Uri.parse('$baseUrl/users/$userId/update').replace(queryParameters: queryParams);

      final response = await _client.post(
        uri,
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        _logger.i('Profile updated successfully');
        final jsonData = json.decode(response.body);
        // Reload user profile to get updated data
        return await getUserProfile(userId);
      } else {
        throw Exception('Failed to update profile: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error updating user profile: $e');
      rethrow;
    }
  }

  /// Get user achievements
  Future<List<Achievement>> getUserAchievements(String userId) async {
    try {
      _logger.i('Fetching achievements for user: $userId');

      final response = await _client.get(
        Uri.parse('$baseUrl/users/$userId/achievements'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> jsonData = json.decode(response.body);
        return jsonData.map((json) => Achievement.fromJson(json)).toList();
      } else {
        throw Exception('Failed to get achievements: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting achievements: $e');
      rethrow;
    }
  }

  /// Get available rewards
  Future<List<Map<String, dynamic>>> getRewardsFromAPI() async {
    try {
      _logger.i('Fetching available rewards from API');

      final response = await _client.get(
        Uri.parse('$baseUrl/rewards'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _logger.i('Loaded ${data['total_count']} rewards');
        return List<Map<String, dynamic>>.from(data['rewards']);
      } else {
        throw Exception('Failed to get rewards: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting rewards: $e');
      rethrow;
    }
  }

  /// Redeem a reward
  Future<Map<String, dynamic>> redeemReward({
    required String userId,
    required String rewardId,
  }) async {
    try {
      _logger.i('Redeeming reward: $rewardId for user: $userId');

      final response = await _client.post(
        Uri.parse('$baseUrl/rewards/redeem').replace(queryParameters: {
          'user_id': userId,
          'reward_id': rewardId,
        }),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _logger.i('Reward redeemed successfully: ${data['message']}');
        return data;
      } else {
        final error = json.decode(response.body);
        throw Exception(error['detail'] ?? 'Failed to redeem reward');
      }
    } catch (e) {
      _logger.e('Error redeeming reward: $e');
      rethrow;
    }
  }

  /// Get leaderboard
  Future<List<Map<String, dynamic>>> getLeaderboard(String period) async {
    try {
      _logger.i('Fetching leaderboard: $period');

      final response = await _client.get(
        Uri.parse('$baseUrl/leaderboard/$period'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final jsonData = json.decode(response.body);
        final List<dynamic> users = jsonData['users'];
        return users.cast<Map<String, dynamic>>();
      } else {
        throw Exception('Failed to get leaderboard: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting leaderboard: $e');
      rethrow;
    }
  }

  /// Get user statistics
  Future<Map<String, dynamic>> getUserStatistics(String userId) async {
    try {
      _logger.i('Fetching statistics for user: $userId');

      final response = await _client.get(
        Uri.parse('$baseUrl/users/$userId/statistics'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Failed to get statistics: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting statistics: $e');
      rethrow;
    }
  }

  /// Get classification history
  Future<List<ClassificationResult>> getClassificationHistory({
    required String userId,
    int limit = 50,
  }) async {
    try {
      _logger.i('Fetching classification history for user: $userId');

      final response = await _client.get(
        Uri.parse('$baseUrl/users/$userId/history?limit=$limit'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final List<dynamic> jsonData = json.decode(response.body);
        return jsonData
            .map((json) => ClassificationResult.fromJson(json))
            .toList();
      } else {
        throw Exception('Failed to get history: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error getting history: $e');
      rethrow;
    }
  }

  /// User authentication - login
  Future<Map<String, dynamic>> login({
    required String email,
    required String password,
  }) async {
    try {
      _logger.i('Logging in user: $email');

      final response = await _client.post(
        Uri.parse('$baseUrl/auth/login'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'email': email,
          'password': password,
        }),
      );

      if (response.statusCode == 200) {
        return json.decode(response.body);
      } else {
        throw Exception('Login failed: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error logging in: $e');
      rethrow;
    }
  }

  /// User authentication - register
  Future<Map<String, dynamic>> register({
    required String name,
    required String email,
    required String password,
    required String neighborhood,
    required String division,
  }) async {
    try {
      _logger.i('Registering user: $email');

      final response = await _client.post(
        Uri.parse('$baseUrl/auth/register'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({
          'name': name,
          'email': email,
          'password': password,
          'neighborhood': neighborhood,
          'division': division,
        }),
      );

      if (response.statusCode == 201) {
        return json.decode(response.body);
      } else {
        throw Exception('Registration failed: ${response.body}');
      }
    } catch (e) {
      _logger.e('Error registering: $e');
      rethrow;
    }
  }

  /// Get redemption history
  Future<List<Map<String, dynamic>>> getRedemptionHistory(String userId) async {
    try {
      _logger.i('Fetching redemption history for user: $userId');
      final response = await _client.get(
        Uri.parse('$baseUrl/rewards/history/$userId'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _logger.i('Loaded ${data['total_redemptions']} redemptions');
        return List<Map<String, dynamic>>.from(data['redemptions']);
      } else {
        throw Exception('Failed to load redemption history: ${response.statusCode}');
      }
    } catch (e) {
      _logger.e('Error fetching redemption history: $e');
      rethrow;
    }
  }

  /// Get all challenges with user progress
  Future<Map<String, dynamic>> getChallenges(String userId) async {
    try {
      _logger.i('Fetching challenges for user: $userId');
      final response = await _client.get(
        Uri.parse('$baseUrl/challenges?user_id=$userId'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _logger.i('Loaded ${data['total_challenges']} challenges');
        return data;
      } else {
        throw Exception('Failed to load challenges: ${response.statusCode}');
      }
    } catch (e) {
      _logger.e('Error fetching challenges: $e');
      rethrow;
    }
  }

  /// Claim challenge reward
  Future<Map<String, dynamic>> claimChallengeReward(String challengeId, String userId) async {
    try {
      _logger.i('Claiming reward for challenge: $challengeId');
      final response = await _client.post(
        Uri.parse('$baseUrl/challenges/$challengeId/claim?user_id=$userId'),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        _logger.i('Claimed ${data['reward']} points');
        return data;
      } else {
        final error = json.decode(response.body);
        throw Exception(error['detail'] ?? 'Failed to claim reward');
      }
    } catch (e) {
      _logger.e('Error claiming challenge reward: $e');
      rethrow;
    }
  }

  /// Dispose the client
  void dispose() {
    _client.close();
  }
}
