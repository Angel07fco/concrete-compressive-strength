from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('modelo_gradient_boosting_jireh.pkl')
scaler = joblib.load('dataFrameScalado_jireh.pkl')

app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request en formato JSON
        data = request.get_json()

        tiempo_de_espera = data['tiempo_espera']
        duracion_de_atencion = data['duracion_atencion']
        costo_servicio = data['costo_servicio']
        calidad_atencion = data['calidad_atencion']
        instalaciones = data['instalaciones']
        comunicacion = data['comunicacion']
        facilidad_de_pago = data['facilidad_de_pago']

        # Crear un DataFrame con todas las características necesarias
        input_data = pd.DataFrame({
            'Tipo_Servicio': [0],
            'Tiempo_Espera': [tiempo_de_espera],
            'Duración_Atención': [duracion_de_atencion],
            'Costo_Servicio': [costo_servicio],
            'Calidad_Atención': [calidad_atencion],
            'Instalaciones': [instalaciones],
            'Comunicación': [comunicacion],
            'Facilidad_Pago': [facilidad_de_pago],
            'Experiencia_Cliente': [0],
            'Recomendaciones': [0],
        })
        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [1,2,3,4,5,6,7]]  # Asegúrate de que estos índices son correctos
        print(scaled_data_for_prediction)

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)

        # Devolver la predicción como JSON
        prediction_value = (prediccion[0])  # Convertir a int para que sea JSON serializable
        return jsonify({'prediction': prediction_value})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)