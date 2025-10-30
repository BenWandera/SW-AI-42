import 'package:flutter/material.dart';

class RecyclingCentersScreen extends StatefulWidget {
  const RecyclingCentersScreen({super.key});

  @override
  State<RecyclingCentersScreen> createState() => _RecyclingCentersScreenState();
}

class _RecyclingCentersScreenState extends State<RecyclingCentersScreen> {
  final List<Map<String, dynamic>> _recyclingCenters = [
    {
      'name': 'Yo-Waste Innovation Hub',
      'address': 'Plot 10, Ntinda Complex, Ntinda Road, Kampala',
      'distance': '4.8 km',
      'phone': '+256 772 345 678',
      'hours': 'Mon-Fri: 8:00AM-5:00PM, Sat: 9:00AM-1:00PM',
      'types': ['Plastic', 'Paper', 'Glass', 'Metal'],
      'rating': 4.6,
    },
    {
      'name': 'KCCA Kiteezi Landfill & Recycling',
      'address': 'Kiteezi, Wakiso District (20km from city center)',
      'distance': '20.3 km',
      'phone': '+256 417 251 000',
      'hours': 'Mon-Sat: 7:00AM-6:00PM',
      'types': ['Plastic', 'Paper', 'Glass', 'Metal', 'Organic'],
      'rating': 3.9,
    },
    {
      'name': 'Eco Brixs Uganda',
      'address': 'Plot 2445, Mukwano Road, Ntinda Industrial Area',
      'distance': '5.2 km',
      'phone': '+256 752 890 143',
      'hours': 'Mon-Fri: 8:00AM-5:00PM',
      'types': ['Plastic'],
      'rating': 4.7,
    },
    {
      'name': 'Takataka Plastics Limited',
      'address': 'Plot 11, Bugolobi, Spring Road, Kampala',
      'distance': '3.5 km',
      'phone': '+256 393 194 380',
      'hours': 'Mon-Fri: 8:00AM-5:00PM',
      'types': ['Plastic'],
      'rating': 4.5,
    },
    {
      'name': 'Bees & Trees Uganda',
      'address': 'Bukoto Street, Plot 21, Kampala',
      'distance': '6.1 km',
      'phone': '+256 701 555 123',
      'hours': 'Mon-Sat: 8:00AM-6:00PM',
      'types': ['Paper', 'Organic', 'Textile'],
      'rating': 4.4,
    },
    {
      'name': 'WEID Centre (E-Waste)',
      'address': 'Plot 1782, Kyaggwe Road, Nakawa Industrial Area',
      'distance': '7.8 km',
      'phone': '+256 772 468 135',
      'hours': 'Mon-Fri: 9:00AM-5:00PM',
      'types': ['E-Waste', 'Batteries', 'Electronic'],
      'rating': 4.8,
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF5F7FA),
      body: CustomScrollView(
        slivers: [
          // Modern Header
          SliverAppBar(
            expandedHeight: 200,
            pinned: true,
            elevation: 0,
            backgroundColor: Colors.transparent,
            leading: IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.white),
              onPressed: () => Navigator.pop(context),
            ),
            flexibleSpace: FlexibleSpaceBar(
              centerTitle: false,
              titlePadding: const EdgeInsets.only(left: 60, bottom: 16),
              title: const Text(
                'Recycling Centers',
                style: TextStyle(
                  fontWeight: FontWeight.w900,
                  fontSize: 22,
                  color: Colors.white,
                  letterSpacing: 0.5,
                ),
              ),
              background: Container(
                decoration: const BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [
                      Color(0xFF2196F3),
                      Color(0xFF42A5F5),
                      Color(0xFF64B5F6),
                    ],
                  ),
                ),
                child: Stack(
                  children: [
                    // Pattern
                    Positioned.fill(
                      child: CustomPaint(
                        painter: _MapPatternPainter(),
                      ),
                    ),
                    // Icon
                    SafeArea(
                      child: Center(
                        child: Container(
                          margin: const EdgeInsets.only(top: 20),
                          padding: const EdgeInsets.all(20),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white.withOpacity(0.2),
                          ),
                          child: const Icon(
                            Icons.location_on_rounded,
                            size: 50,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Content
          SliverPadding(
            padding: const EdgeInsets.all(16),
            sliver: SliverList(
              delegate: SliverChildListDelegate([
                // Info Card
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF4CAF50), Color(0xFF66BB6A)],
                    ),
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.info_outline, color: Colors.white, size: 24),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'Found ${_recyclingCenters.length} centers near you',
                          style: const TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),

                // Centers List
                ..._recyclingCenters.map((center) => _buildCenterCard(center)).toList(),
              ]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCenterCard(Map<String, dynamic> center) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: () {
            // Show details or open map
          },
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF2196F3), Color(0xFF1976D2)],
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(Icons.recycling_rounded, color: Colors.white, size: 24),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            center['name'],
                            style: const TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          Row(
                            children: [
                              const Icon(Icons.star, size: 14, color: Colors.amber),
                              const SizedBox(width: 4),
                              Text(
                                '${center['rating']}',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.grey[600],
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: const Color(0xFF4CAF50).withOpacity(0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.location_on, size: 14, color: Color(0xFF4CAF50)),
                          const SizedBox(width: 4),
                          Text(
                            center['distance'],
                            style: const TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: Color(0xFF4CAF50),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                // Address
                Row(
                  children: [
                    Icon(Icons.place_rounded, size: 16, color: Colors.grey[600]),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        center['address'],
                        style: TextStyle(
                          fontSize: 13,
                          color: Colors.grey[700],
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),

                // Phone
                Row(
                  children: [
                    Icon(Icons.phone_rounded, size: 16, color: Colors.grey[600]),
                    const SizedBox(width: 8),
                    Text(
                      center['phone'],
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey[700],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),

                // Hours
                Row(
                  children: [
                    Icon(Icons.access_time_rounded, size: 16, color: Colors.grey[600]),
                    const SizedBox(width: 8),
                    Text(
                      center['hours'],
                      style: TextStyle(
                        fontSize: 13,
                        color: Colors.grey[700],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                // Accepted Types
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: (center['types'] as List<String>).map((type) {
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(
                        color: _getTypeColor(type).withOpacity(0.15),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        type,
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: _getTypeColor(type),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Color _getTypeColor(String type) {
    switch (type) {
      case 'Plastic':
        return const Color(0xFF2196F3);
      case 'Paper':
        return const Color(0xFFFF9800);
      case 'Glass':
        return const Color(0xFF4CAF50);
      case 'Metal':
        return const Color(0xFF9E9E9E);
      case 'E-Waste':
        return const Color(0xFF9C27B0);
      case 'Organic':
        return const Color(0xFF8BC34A);
      case 'Batteries':
        return const Color(0xFFFFEB3B);
      case 'Hazardous':
        return const Color(0xFFF44336);
      case 'Textile':
        return const Color(0xFFE91E63);
      default:
        return Colors.grey;
    }
  }
}

class _MapPatternPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.08)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    // Draw location pins
    for (double x = 0; x < size.width; x += 80) {
      for (double y = 0; y < size.height; y += 80) {
        _drawPin(canvas, Offset(x, y), paint);
      }
    }
  }

  void _drawPin(Canvas canvas, Offset center, Paint paint) {
    final path = Path();
    path.moveTo(center.dx, center.dy - 10);
    path.lineTo(center.dx - 5, center.dy - 15);
    path.lineTo(center.dx + 5, center.dy - 15);
    path.close();
    canvas.drawPath(path, paint);
    canvas.drawCircle(Offset(center.dx, center.dy - 15), 3, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
