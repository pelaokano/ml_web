from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import pandas as pd
app = Flask(__name__)

etiquetas = {0: "adelante", 1: "derecha", 2: "izquierda", 3: "atras"}

modelo = tf.keras.models.load_model('modelos/model_150ms.h5')

@app.route("/predict", methods=["POST"])
def predict():
    req_json = request.get_json()

    data = np.array(req_json["data"])
    
    if data.shape[0] % 15 == 0:
        data = data.reshape(int(data.shape[0]/15), 15, 3)
        prediccion = modelo.predict(data)
        prediccion = [etiquetas[np.argmax(o)] for o in prediccion]
    else:
        print("error: la cantidad de registros no es divisible por 15")
        prediccion = ["error: la cantidad de registros no es divisible por 15"]

    responses = jsonify(prueba = prediccion)
    responses.status_code = 200

    return (responses)

if __name__ == "__main__":

    app.run(host="127.0.0.1", port=8000, debug=True)
