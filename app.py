from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado
model = joblib.load('concrete_modelo.pkl')
scaler = joblib.load('concrete_escalado.pkl')
app.logger.debug('Modelo cargado correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        cement = float(request.form['cement'])
        blast = float(request.form['blast'])
        water =float(request.form['water'])
        superplasticizer =float(request.form['superplasticizer'])
        age =float(request.form['age'])

        input_data = pd.DataFrame({
            'Cement': [cement],
            'Blast Furnace Slag': [blast],
            'Fly Ash': [0],
            'Water': [water],
            'Superplasticizer': [superplasticizer],
            'Coarse Aggregate': [0],
            'Fine Aggregate': [0],
            'Age (day)': [age]
        })

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [0,1,2,3,7]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = round(float(prediccion[0]), 2)

        return jsonify({'prediction': prediction_value})

    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)