from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import datetime
from roi_resell import PropertyValuePredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize predictor
predictor = PropertyValuePredictor()

@app.route("/")
def home():
    return jsonify({"status": "success", "message": "App is running!"})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for ML + ROI prediction"""
    try:
        data = request.json

        # Validate required fields
        required_fields = ['city', 'sub_location', 'bedroom', 'status', 'carpet_area', 'total_area']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Load models
        if predictor.price_model is None:
            if not predictor.load_models():
                return jsonify({'error': 'Models not loaded'}), 500

        # Predict
        years = data.get('years', 5)
        result = predictor.predict_property_value(data, years)

        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['GET'])
def train_models():
    """Train price and rent models from CSV"""
    try:
        csv_file = 'cleaned_resale_properties.csv'
        if not os.path.exists(csv_file):
            return jsonify({'status': 'error', 'message': f'File "{csv_file}" not found'}), 404

        success = predictor.train_models(csv_file)

        if success:
            return jsonify({'status': 'success', 'message': 'Models trained successfully'})
        else:
            return jsonify({'status': 'error', 'message': 'Training failed'}), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/check_models', methods=['GET'])
def check_models():
    """Check if trained models exist"""
    return jsonify({
        'models_loaded': predictor.price_model is not None,
        'models_exist': os.path.exists('models/price_model.pkl')
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model metadata and available encodings"""
    try:
        if predictor.price_model is None:
            predictor.load_models()

        return jsonify({
            'models_loaded': predictor.price_model is not None,
            'location_multipliers': predictor.location_multipliers,
            'bedroom_multipliers': predictor.bedroom_multipliers,
            'rental_yields': predictor.rental_yields,
            'supported_cities': list(predictor.rental_yields.keys()) if predictor.city_encoder else [],
            'feature_columns': predictor.feature_columns if predictor.feature_columns else []
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_calculation', methods=['POST'])
def save_calculation():
    """Save JSON result of prediction"""
    try:
        data = request.json
        os.makedirs('calculations', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'calculations/roi_calculation_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return jsonify({
            'status': 'success',
            'message': 'Calculation saved',
            'filename': filename
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    try:
        predictor.load_models()
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️  Models not loaded: {e}")
        print("💡 Train models via /train")

    app.run(debug=True, host='0.0.0.0', port=5000)
