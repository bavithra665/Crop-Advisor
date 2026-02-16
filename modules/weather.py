import requests
import os

class ClimateRiskEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather_data(self, city):
        if not self.api_key:
            return None
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            response = requests.get(self.base_url, params=params)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            print(f"Weather API Error: {e}")
            return None

    def calculate_risk_scores(self, city, hist_rainfall, hist_temp):
        """
        Calculates drought and flood risk scores (0-100).
        """
        if not city:
            weather = None
        else:
            weather = self.get_weather_data(city)
        # Mocking data if API fails for demo purposes
        temp = weather['main']['temp'] if weather else hist_temp
        humidity = weather['main']['humidity'] if weather else 50
        
        # Drought Risk Logic
        # - High temp, Low rainfall history, Low humidity
        drought_score = (max(0, temp - 25) * 2) + (max(0, 100 - humidity) * 0.5)
        if hist_rainfall < 500:
            drought_score += 20
        
        # Flood Risk Logic
        # - High rainfall history, current humidity
        flood_score = (hist_rainfall / 50) + (humidity * 0.3)
        
        return {
            'drought_risk': min(100, round(drought_score)),
            'flood_risk': min(100, round(flood_score)),
            'current_temp': temp,
            'current_humidity': humidity
        }

    def get_risk_adjusted_crops(self, predictions, risk_scores):
        """
        Adjusts crop recommendations based on risk.
        If drought risk is high, prioritize drought-resistant crops (Millets).
        If flood risk is high, prioritize water-loving crops (Rice).
        """
        d_risk = risk_scores['drought_risk']
        f_risk = risk_scores['flood_risk']
        
        adjusted = []
        for crop in predictions:
            score_adj = 0
            if d_risk > 60:
                if crop['name'].lower() in ['millets', 'pulses', 'maize']:
                    score_adj = 15
                elif crop['name'].lower() in ['rice', 'cotton']:
                    score_adj = -20
            
            if f_risk > 70:
                if crop['name'].lower() in ['rice']:
                    score_adj = 20
                elif crop['name'].lower() in ['pulses', 'maize']:
                    score_adj = -25
            
            crop['risk_adjusted_confidence'] = max(0, min(100, crop['confidence'] + score_adj))
            adjusted.append(crop)
            
        return sorted(adjusted, key=lambda x: x['risk_adjusted_confidence'], reverse=True)
