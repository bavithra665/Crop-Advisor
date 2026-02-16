"""
Lightweight rule-based crop predictor for cloud deployment
Works without the 23MB ML model file
"""

def predict_crops_simple(n, p, k, temperature, humidity, ph, rainfall, soil_type, season, region):
    """
    Simple rule-based prediction when ML model unavailable
    Returns top 3 crops with confidence scores
    """
    
    # Crop database with requirements
    crops_db = {
        'Rice': {
            'n_range': (80, 120), 'p_range': (40, 60), 'k_range': (40, 60),
            'temp_range': (20, 35), 'rainfall_min': 1000, 'ph_range': (5.5, 7.0),
            'soils': ['Clayey', 'Loamy'], 'seasons': ['Kharif', 'Monsoon']
        },
        'Wheat': {
            'n_range': (100, 140), 'p_range': (40, 80), 'k_range': (40, 80),
            'temp_range': (15, 25), 'rainfall_min': 500, 'ph_range': (6.0, 7.5),
            'soils': ['Loamy', 'Clayey'], 'seasons': ['Rabi', 'Winter']
        },
        'Maize': {
            'n_range': (60, 100), 'p_range': (30, 60), 'k_range': (30, 60),
            'temp_range': (20, 30), 'rainfall_min': 600, 'ph_range': (5.5, 7.5),
            'soils': ['Loamy', 'Sandy', 'Black'], 'seasons': ['Kharif', 'Summer']
        },
        'Cotton': {
            'n_range': (80, 120), 'p_range': (40, 80), 'k_range': (40, 80),
            'temp_range': (21, 35), 'rainfall_min': 600, 'ph_range': (6.0, 8.0),
            'soils': ['Black', 'Alluvial'], 'seasons': ['Kharif', 'Summer']
        },
        'Millets': {
            'n_range': (40, 80), 'p_range': (20, 40), 'k_range': (20, 40),
            'temp_range': (25, 35), 'rainfall_min': 300, 'ph_range': (5.0, 7.5),
            'soils': ['Sandy', 'Red', 'Loamy'], 'seasons': ['Kharif', 'Summer']
        },
        'Pulses': {
            'n_range': (20, 60), 'p_range': (40, 80), 'k_range': (20, 60),
            'temp_range': (20, 30), 'rainfall_min': 400, 'ph_range': (6.0, 7.5),
            'soils': ['Loamy', 'Black', 'Red'], 'seasons': ['Rabi', 'Winter']
        },
        'Sugarcane': {
            'n_range': (80, 150), 'p_range': (40, 80), 'k_range': (80, 150),
            'temp_range': (21, 35), 'rainfall_min': 1000, 'ph_range': (6.0, 7.5),
            'soils': ['Loamy', 'Black'], 'seasons': ['Whole Year', 'Monsoon']
        },
        'Jute': {
            'n_range': (60, 100), 'p_range': (30, 60), 'k_range': (30, 60),
            'temp_range': (24, 35), 'rainfall_min': 1200, 'ph_range': (6.0, 7.5),
            'soils': ['Alluvial', 'Clayey'], 'seasons': ['Kharif', 'Monsoon']
        },
    }
    
    scores = {}
    
    for crop, req in crops_db.items():
        score = 0
        
        # NPK matching (40 points)
        if req['n_range'][0] <= n <= req['n_range'][1]:
            score += 15
        if req['p_range'][0] <= p <= req['p_range'][1]:
            score += 12
        if req['k_range'][0] <= k <= req['k_range'][1]:
            score += 13
            
        # Temperature (20 points)
        if req['temp_range'][0] <= temperature <= req['temp_range'][1]:
            score += 20
        elif abs(temperature - sum(req['temp_range'])/2) < 5:
            score += 10
            
        # Rainfall (15 points)
        if rainfall >= req['rainfall_min']:
            score += 15
        elif rainfall >= req['rainfall_min'] * 0.7:
            score += 8
            
        # pH (10 points)
        if req['ph_range'][0] <= ph <= req['ph_range'][1]:
            score += 10
        elif abs(ph - sum(req['ph_range'])/2) < 0.5:
            score += 5
            
        # Soil type (10 points)
        if soil_type in req['soils']:
            score += 10
            
        # Season (5 points)
        if season in req['seasons']:
            score += 5
            
        scores[crop] = min(100, score)
    
    # Get top 3
    sorted_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    return [
        {'name': crop, 'confidence': round(conf, 2)}
        for crop, conf in sorted_crops
    ]
