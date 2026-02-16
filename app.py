from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os
from dotenv import load_dotenv

# Import Custom Modules
# (Moved into lazy getters to speed up startup)

# Load environment variables
load_dotenv()

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'crop-advisor-secret-key-2026')

# ---------------- SQLite Setup ----------------
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'crop_advisor.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- Health Check ----------------
@app.route('/health')
def health_check():
    return "OK", 200

# ---------------- Global Engine Holders ----------------
_risk_engine = None
_agri_bot = None
_analytics_engine = None

def get_risk_engine():
    global _risk_engine
    if _risk_engine is None:
        try:
            from modules.weather import ClimateRiskEngine
            _risk_engine = ClimateRiskEngine()
            print("✅ Climate Risk Engine initialized")
        except Exception as e:
            print(f"⚠️ Climate Risk Engine failed: {e}")
    return _risk_engine

def get_agri_bot():
    global _agri_bot
    if _agri_bot is None:
        try:
            from modules.chatbot import AgriBot
            _agri_bot = AgriBot()
            print("✅ AgriBot initialized")
        except Exception as e:
            print(f"⚠️ AgriBot failed: {e}")
    return _agri_bot

def get_analytics_engine():
    global _analytics_engine
    if _analytics_engine is None:
        try:
            from modules.analytics import AnalyticsEngine
            _analytics_engine = AnalyticsEngine()
            print("✅ Analytics Engine initialized")
        except Exception as e:
            print(f"⚠️ Analytics Engine failed: {e}")
    return _analytics_engine

# ---------------- ML Model Getter ----------------
_model_bundle = None

def get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        try:
            import joblib
            model_path = os.path.join(os.path.dirname(__file__), 'models/crop_model.pkl')
            _model_bundle = joblib.load(model_path)
            print("✅ ML Model loaded successfully")
        except Exception as e:
            print(f"❌ CRITICAL: ML Model failed to load: {e}")
            _model_bundle = {}
    return _model_bundle

# ---------------- MongoDB Connection Getter ----------------
_crop_collection = None

def get_crop_collection():
    global _crop_collection
    if _crop_collection is None:
        try:
            from pymongo import MongoClient
            mongo_uri = os.environ.get('MONGODB_URI', "mongodb://localhost:27017")
            mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            mongo_db = mongo_client["agri_predictor_db"]
            _crop_collection = mongo_db["crops"]
            # Test connection
            mongo_client.admin.command('ping')
            print("✅ MongoDB connected!")
        except Exception as e:
            print("⚠️ MongoDB connection error:", e)
            _crop_collection = None
    return _crop_collection

# ---------------- SQLAlchemy Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    location = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    n = db.Column(db.Float, nullable=False)
    p = db.Column(db.Float, nullable=False)
    k = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)
    soil_type = db.Column(db.String(50), nullable=False)
    season = db.Column(db.String(50), nullable=False)
    region = db.Column(db.String(50), nullable=False)
    crop1 = db.Column(db.String(50), nullable=False)
    confidence1 = db.Column(db.Float, nullable=False)
    crop2 = db.Column(db.String(50), nullable=False)
    confidence2 = db.Column(db.Float, nullable=False)
    crop3 = db.Column(db.String(50), nullable=False)
    confidence3 = db.Column(db.Float, nullable=False)
    
    # New analytics fields
    drought_risk = db.Column(db.Float, default=0)
    flood_risk = db.Column(db.Float, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='feedbacks')

# Initialize DB on first request instead of startup
@app.before_request
def init_db():
    if not hasattr(app, 'db_initialized'):
        try:
            db.create_all()
            app.db_initialized = True
            print("✅ Database initialized")
        except Exception as e:
            print(f"⚠️ Initial DB setup skipped or failed: {e}")

# Import numpy where needed
def get_numpy():
    import numpy as np
    return np

# ---------------- Local Crop Details (fallback) ----------------
crop_details = {
    'Rice': {'planting': 'June-July', 'fertilizer': 'Urea, DAP, MOP', 'irrigation': 'Flooded field method', 'yield': '4-6 tons/ha', 'image': 'rice.jpg'},
    'Wheat': {'planting': 'October-November', 'fertilizer': 'NPK 20:20:20', 'irrigation': 'Sprinkler/Furrow', 'yield': '4-5 tons/ha', 'image': 'wheat.jpg'},
    'Maize': {'planting': 'June-July or Feb-March', 'fertilizer': 'Urea, DAP', 'irrigation': 'Drip/Sprinkler', 'yield': '6-8 tons/ha', 'image': 'maize.jpg'},
    'Millets': {'planting': 'June-August', 'fertilizer': 'NPK 10:10:10', 'irrigation': 'Moderate', 'yield': '2-4 tons/ha', 'image': 'millets.jpg'},
    'Pulses': {'planting': 'July-August', 'fertilizer': 'DAP, Urea', 'irrigation': 'Low', 'yield': '1-2 tons/ha', 'image': 'pulses.jpg'},
    'Cotton': {'planting': 'April-May', 'fertilizer': 'NPK 20:20:20', 'irrigation': 'Moderate', 'yield': '2-3 tons/ha', 'image': 'cotton.jpg'},
    'Coffee': {'planting': 'June-August', 'fertilizer': 'NPK 10:5:20', 'irrigation': 'Drip/Sprinkler', 'yield': '1-2 tons/ha', 'image': 'coffee.jpg'},
    'Jute': {'planting': 'March-May', 'fertilizer': 'N:P:K 2:1:1', 'irrigation': 'Rainfed/Flooded', 'yield': '2-3 tons/ha', 'image': 'jute.jpg'},
    'Tea': {'planting': 'June-September', 'fertilizer': 'Ammonium Sulphate', 'irrigation': 'Sprinkler', 'yield': '2-3 tons/ha', 'image': 'tea.jpg'},
    'Sugarcane': {'planting': 'Feb-March', 'fertilizer': '250:125:125 NPK kg/ha', 'irrigation': 'Furrow', 'yield': '80-100 tons/ha', 'image': 'sugarcane.jpg'},
    'Tobacco': {'planting': 'August-October', 'fertilizer': 'NPK 50:50:50', 'irrigation': 'Furrow', 'yield': '2-3 tons/ha', 'image': 'tobacco.jpg'},
    'Rubber': {'planting': 'June-July', 'fertilizer': 'NPK 10:10:4', 'irrigation': 'Rainfed', 'yield': '1-2 tons/ha', 'image': 'rubber.jpg'},
    'Coconut': {'planting': 'May-June', 'fertilizer': 'NPK 500:320:1200g/palm', 'irrigation': 'Drip/Basin', 'yield': '80-100 nuts/palm', 'image': 'coconut.jpg'},
    'Banana': {'planting': 'Feb-April', 'fertilizer': 'NPK 200:50:200g/plant', 'irrigation': 'Drip', 'yield': '30-40 tons/ha', 'image': 'banana.jpg'},
    'Grapes': {'planting': 'Oct-Jan', 'fertilizer': 'FYM + NPK', 'irrigation': 'Drip', 'yield': '20-30 tons/ha', 'image': 'grapes.jpg'},
    'Apple': {'planting': 'Jan-Feb', 'fertilizer': 'FYM + NPK', 'irrigation': 'Drip', 'yield': '10-15 tons/ha', 'image': 'apple.jpg'},
    'Mango': {'planting': 'July-Aug', 'fertilizer': 'FYM + 1kg NPK/tree', 'irrigation': 'Basin', 'yield': '8-10 tons/ha', 'image': 'mango.jpg'},
    'Muskmelon': {'planting': 'Feb-March', 'fertilizer': 'NPK 100:50:50', 'irrigation': 'Drip', 'yield': '15-20 tons/ha', 'image': 'muskmelon.jpg'},
    'Watermelon': {'planting': 'Jan-March', 'fertilizer': 'NPK 100:50:50', 'irrigation': 'Drip', 'yield': '20-25 tons/ha', 'image': 'watermelon.jpg'},
    'Orange': {'planting': 'July-Aug', 'fertilizer': 'NPK 600:200:300g/tree', 'irrigation': 'Drip/Basin', 'yield': '10-12 tons/ha', 'image': 'orange.jpg'},
    'Papaya': {'planting': 'Feb-March or June-July', 'fertilizer': 'NPK 200:200:250g/plant', 'irrigation': 'Drip', 'yield': '30-40 tons/ha', 'image': 'papaya.jpg'},
    'Pomegranate': {'planting': 'Feb-March', 'fertilizer': 'FYM + NPK', 'irrigation': 'Drip', 'yield': '10-12 tons/ha', 'image': 'pomegranate.jpg'},
    'Lentil': {'planting': 'Oct-Nov', 'fertilizer': 'DAP + Sulphur', 'irrigation': 'Rainfed/Light', 'yield': '1-1.5 tons/ha', 'image': 'lentil.jpg'},
    'Blackgram': {'planting': 'Feb-March or June-July', 'fertilizer': 'DAP', 'irrigation': 'Rainfed', 'yield': '0.8-1 tons/ha', 'image': 'blackgram.jpg'},
    'Mungbean': {'planting': 'Feb-March or June-July', 'fertilizer': 'DAP', 'irrigation': 'Rainfed', 'yield': '0.8-1 tons/ha', 'image': 'mungbean.jpg'},
    'Mothbeans': {'planting': 'June-July', 'fertilizer': 'FYM', 'irrigation': 'Rainfed', 'yield': '0.4-0.6 tons/ha', 'image': 'mothbeans.jpg'},
    'Pigeonpeas': {'planting': 'June-July', 'fertilizer': 'DAP + FYM', 'irrigation': 'Rainfed', 'yield': '1.5-2 tons/ha', 'image': 'pigeonpeas.jpg'},
    'Kidneybeans': {'planting': 'May-June', 'fertilizer': 'NPK 40:60:40', 'irrigation': 'Rainfed', 'yield': '1.0-1.5 tons/ha', 'image': 'kidneybeans.jpg'},
    'Chickpea': {'planting': 'Oct-Nov', 'fertilizer': 'DAP', 'irrigation': 'Sprinkler', 'yield': '1.5-2 tons/ha', 'image': 'chickpea.jpg'}
}

# ---------------- Utility: Get Image File Case-Insensitive ----------------
def get_image_filename(image_name):
    folder = os.path.join(app.static_folder, 'images')
    if not os.path.exists(folder):
        return 'default.jpg'
    base_name = image_name.lower()
    if not base_name.endswith('.jpg'):
        base_name += '.jpg'
    for f in os.listdir(folder):
        if f.lower() == base_name:
            return f
    return 'default.jpg'

# ---------------- Routes ----------------
@app.route('/health')
def health():
    """Health check endpoint for deployment platforms"""
    return jsonify({"status": "healthy", "service": "AgriPredictor-AI"}), 200

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        location = request.form.get('location')
        password = request.form.get('password')
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please login.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        new_user = User(name=name, email=email, location=location, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('home'))

@app.route('/predictcrop', methods=['GET', 'POST'])
def predictcrop():
    if 'user_id' not in session:
        flash('Please login to access crop prediction.', 'warning')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])

    if request.method == 'POST':
        try:
            n = float(request.form.get('n'))
            p = float(request.form.get('p'))
            k = float(request.form.get('k'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            ph = float(request.form.get('ph'))
            rainfall = float(request.form.get('rainfall'))
            soil_type = request.form.get('soil_type')
            season = request.form.get('season')
            region = request.form.get('region')

            # Load Model Bundle (Lazy)
            bundle = get_model_bundle()
            if not bundle:
                flash("AI Prediction system is warming up. Please try again in a few seconds.", "info")
                return redirect(url_for('predictcrop'))

            model = bundle.get('model')
            le_soil = bundle.get('le_soil')
            le_season = bundle.get('le_season')
            le_region = bundle.get('le_region')
            le_crop = bundle.get('le_crop')

            # Build Model Features
            soil_encoded = le_soil.transform([soil_type])[0]
            season_encoded = le_season.transform([season])[0]
            region_encoded = le_region.transform([region])[0]

            np = get_numpy()
            features = np.array([[n, p, k, temperature, humidity, ph, rainfall,
                                  soil_encoded, season_encoded, region_encoded]])

            probabilities = model.predict_proba(features)[0]
            top_3_indices = np.argsort(probabilities)[-3:][::-1]

            top_3_crops = []
            crop_col = get_crop_collection()
            
            for idx in top_3_indices:
                crop_name = le_crop.inverse_transform([idx])[0]
                confidence = round(probabilities[idx] * 100, 2)

                # Fetch Details (Hybrid Mongo/Local)
                mongo_crop = None
                if crop_col is not None:
                    mongo_crop = crop_col.find_one({"name": crop_name})

                if mongo_crop:
                    details = {
                        'planting': mongo_crop.get('planting', 'N/A'),
                        'fertilizer': mongo_crop.get('fertilizer', 'N/A'),
                        'irrigation': mongo_crop.get('irrigation', 'N/A'),
                        'yield': mongo_crop.get('yield', 'N/A'),
                        'image': get_image_filename(mongo_crop.get('image', 'default.jpg'))
                    }
                else:
                    local_crop = crop_details.get(crop_name, {'image': 'default.jpg'})
                    details = {
                        'planting': local_crop.get('planting', 'N/A'),
                        'fertilizer': local_crop.get('fertilizer', 'N/A'),
                        'irrigation': local_crop.get('irrigation', 'N/A'),
                        'yield': local_crop.get('yield', 'N/A'),
                        'image': get_image_filename(local_crop.get('image', 'default.jpg'))
                    }

                top_3_crops.append({'name': crop_name, 'confidence': confidence, **details})

            # --- New: Climate Risk Adjusted Recommendations ---
            user_location = user.location if user and user.location else "Unknown"
            risk_data = None
            risk_engine = get_risk_engine()
            if risk_engine:
                risk_data = risk_engine.calculate_risk_scores(user_location, rainfall, temperature)
            
            # Add these fallbacks to ensure dictionary keys exist even if weather API fails
            if not risk_data or 'drought_risk' not in risk_data:
                risk_data = {
                    'drought_risk': 20.0, # Safe default
                    'flood_risk': 20.0,   # Safe default
                    'current_temp': temperature,
                    'current_humidity': humidity
                }
            
            # Map adjusted confidence back to main confidence for display simplicity
            # Logic: We keep original ML confidence but store risk metrics
            # Or we can swap them. For now, let's keep ML strict but warn about risk.
            
            adjusted_crops = []
            for crop in top_3_crops:
                # Simple pass-through for now as risk logic is display-only
                adjusted_crops.append(crop)

            # Save to SQLite
            new_pred = Prediction(
                user_id=session['user_id'], n=n, p=p, k=k,
                temperature=temperature, humidity=humidity, ph=ph,
                rainfall=rainfall, soil_type=soil_type,
                season=season, region=region,
                crop1=adjusted_crops[0]['name'], confidence1=adjusted_crops[0]['confidence'],
                crop2=adjusted_crops[1]['name'], confidence2=adjusted_crops[1]['confidence'],
                crop3=adjusted_crops[2]['name'], confidence3=adjusted_crops[2]['confidence'],
                drought_risk=risk_data['drought_risk'],
                flood_risk=risk_data['flood_risk']
            )
            db.session.add(new_pred)
            db.session.commit()

            return render_template('predictcrop.html',
                                   predictions=adjusted_crops,
                                   risk_data=risk_data,
                                   show_results=True,
                                   soil_types=bundle['le_soil'].classes_,
                                   seasons=bundle['le_season'].classes_,
                                   regions=bundle['le_region'].classes_)
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            return redirect(url_for('predictcrop'))

    # GET Request - Check for last prediction
    last_pred = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).first()
    
    saved_predictions = None
    saved_risk_data = None
    show_saved = False

    # Get model classes for dropdowns (Lazy)
    # Get model classes for dropdowns (Lazy)
    bundle = get_model_bundle()
    if bundle and 'le_soil' in bundle:
        try:
            soil_classes = bundle['le_soil'].classes_.tolist()
            season_classes = bundle['le_season'].classes_.tolist()
            region_classes = bundle['le_region'].classes_.tolist()
        except:
             # Fallback if specific attributes are missing or numpy issues
            soil_classes = ['Alluvial', 'Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
            season_classes = ['Kharif', 'Rabi', 'Zaid']
            region_classes = ['North', 'South', 'East', 'West', 'Central']
    else:
        # Absolute fallback if bundle load fails
        soil_classes = ['Alluvial', 'Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
        season_classes = ['Kharif', 'Rabi', 'Zaid']
        region_classes = ['North', 'South', 'East', 'West', 'Central']

    if last_pred:
        # Reconstruct crop objects for display
        saved_predictions = []
        for i in range(1, 4):
            c_name = getattr(last_pred, f'crop{i}')
            c_conf = getattr(last_pred, f'confidence{i}')
            
            # Fetch details again for display
            details = crop_details.get(c_name, {'image': 'default.jpg'})
            crop_col = get_crop_collection()
            if crop_col is not None:
                mongo_crop = crop_col.find_one({"name": c_name})
                if mongo_crop:
                     details = {
                        'planting': mongo_crop.get('planting', 'N/A'),
                        'fertilizer': mongo_crop.get('fertilizer', 'N/A'),
                        'irrigation': mongo_crop.get('irrigation', 'N/A'),
                        'yield': mongo_crop.get('yield', 'N/A'),
                        'image': get_image_filename(mongo_crop.get('image', 'default.jpg'))
                    }
            
            # Ensure local fallback keys if mongo fails or keys exist
            if 'planting' not in details: details['planting'] = 'N/A'
            if 'fertilizer' not in details: details['fertilizer'] = 'N/A'
            if 'yield' not in details: details['yield'] = 'N/A'
            if 'irrigation' not in details: details['irrigation'] = 'N/A'
            if 'image' not in details: details['image'] = get_image_filename('default.jpg')

            saved_predictions.append({'name': c_name, 'confidence': c_conf, **details})
        
        saved_risk_data = {
            'drought_risk': last_pred.drought_risk,
            'flood_risk': last_pred.flood_risk
        }
        show_saved = True

    return render_template('predictcrop.html', 
                           predictions=saved_predictions, 
                           risk_data=saved_risk_data,
                           show_results=show_saved,
                           soil_types=soil_classes,
                           seasons=season_classes,
                           regions=region_classes)

# ----------------- New AI Assistant Route -----------------
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        user_query = request.form.get('query')
        bot = get_agri_bot()
        if bot:
            response = bot.get_answer(user_query)
        else:
            response = "I'm currently warming up my AI systems. Please try again in a minute!"
        return jsonify({'response': response})
        
    return render_template('chatbot.html')

# ----------------- New Analytics Route -----------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_preds = Prediction.query.filter_by(user_id=session['user_id']).all()
    
    # Process analytics
    dist_data = trend_data = comparison_data = []
    engine = get_analytics_engine()
    if engine:
        dist_data = engine.process_prediction_history(user_preds)
        trend_data = engine.get_trend_data(user_preds)
        comparison_data = engine.get_crop_comparison_data(user_preds)
    
    # Calculate Dashboard Stats
    avg_confidence = 0
    risk_level = "N/A"
    risk_class = "text-muted"
    total_recommendations = 0

    if user_preds:
        # 1. Total Recommendations (count all top 3 crops from all predictions)
        for p in user_preds:
            if p.crop1: total_recommendations += 1
            if p.crop2: total_recommendations += 1
            if p.crop3: total_recommendations += 1
        
        # 2. Average Confidence
        total_conf = sum([p.confidence1 for p in user_preds])
        avg_confidence = round(total_conf / len(user_preds))

        # 3. Risk Level (based on latest prediction's highest risk factor)
        latest_pred = user_preds[-1] # Last item is the most recent
        current_risk = max(latest_pred.drought_risk or 0, latest_pred.flood_risk or 0)
        
        if current_risk < 30:
            risk_level = "LOW"
            risk_class = "text-success"
        elif current_risk < 70:
            risk_level = "MODERATE"
            risk_class = "text-warning"
        else:
            risk_level = "HIGH"
            risk_class = "text-danger"
    
    return render_template('dashboard.html', 
                          predictions=user_preds[::-1],
                          total_recommendations=total_recommendations,
                          avg_confidence=avg_confidence,
                          risk_level=risk_level,
                          risk_class=risk_class,
                          dist_data=dist_data,
                          trend_data=trend_data,
                          comparison_data=comparison_data)

@app.route('/review', methods=['GET', 'POST'])
def review():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        feedback = Feedback(user_id=session['user_id'], 
                           rating=int(request.form.get('rating')), 
                           comments=request.form.get('comments'))
        db.session.add(feedback)
        db.session.commit()
        flash('Feedback submitted!', 'success')
    return render_template('review.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
