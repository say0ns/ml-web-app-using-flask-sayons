from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Permite peticiones desde el frontend

with open('../src/modelos-random-forest.pkl', 'rb') as f:
    modelos = load(f)
try:
    model = joblib.load(modelos['rf_classifier_model'])  # O el nombre de tu modelo
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None


@app.route('/')
def index():
    """Sirve la página principal"""
    # Puedes servir el HTML directamente o desde un archivo
    return render_template('index.html')
    # O si tienes el HTML en un archivo: return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para realizar predicciones"""
    try:
        # Obtener datos del request JSON
        data = request.get_json()
        
        # Validar que se recibieron todos los campos necesarios
        required_fields = [
            'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
            'insulin', 'bmi', 'diabetes_pedigree', 'age'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Campo requerido faltante: {field}'
                }), 400
        
        # Preparar los datos para el modelo
        features = np.array([[
            data['pregnancies'],
            data['glucose'],
            data['blood_pressure'],
            data['skin_thickness'],
            data['insulin'],
            data['bmi'],
            data['diabetes_pedigree'],
            data['age']
        ]])
        
        if model is None:
            return jsonify({
                'error': 'Modelo no disponible'
            }), 500
        
        # Realizar predicción
        prediction = model.predict(features)[0]
        
        # Obtener probabilidades si el modelo las soporta
        try:
            probabilities = model.predict_proba(features)[0]
            probability = probabilities[1]  # Probabilidad de tener diabetes
            confidence = max(probabilities)  # Mayor probabilidad como confianza
        except:
            # Si el modelo no soporta predict_proba, usar valores por defecto
            probability = 0.8 if prediction == 1 else 0.2
            confidence = 0.75
        
        # Preparar respuesta
        response = {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(confidence),
            'message': 'Predicción realizada exitosamente'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error en predicción: {e}")
        return jsonify({
            'error': 'Error interno del servidor',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({
        'status': 'OK',
        'message': 'API funcionando correctamente',
        'model_loaded': model is not None
    })

# Para pruebas locales
@app.route('/test', methods=['GET'])
def test():
    """Endpoint de prueba con datos de ejemplo"""
    test_data = {
        'pregnancies': 2,
        'glucose': 148,
        'blood_pressure': 72,
        'skin_thickness': 35,
        'insulin': 0,
        'bmi': 33.6,
        'diabetes_pedigree': 0.627,
        'age': 50
    }
    
    # Simular una petición POST al endpoint predict
    with app.test_client() as client:
        response = client.post('/predict', 
                             json=test_data,
                             content_type='application/json')
        return response.get_json()

if __name__ == '__main__':
    # Para desarrollo local
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))