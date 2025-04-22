# app.py
from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle
import os
import numpy as np
from alertas import enviar_alerta  # 👈 Agregado
import time  # 👈 Agregado para control de tiempo

app = Flask(__name__)

datos_entrenados = pickle.load(open("rostros_entrenados.pkl", "rb"))
rostros_conocidos = datos_entrenados["rostros"]
nombres_conocidos = datos_entrenados["nombres"]

camara = cv2.VideoCapture(0)

ultimo_envio = 0  # 👈 Timestamp del último correo
espera_alerta = 30  # segundos entre alertas

def generar_frames():
    global ultimo_envio
    while True:
        success, frame = camara.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb_frame)
        codificaciones = face_recognition.face_encodings(rgb_frame, ubicaciones)

        for (top, right, bottom, left), codificacion in zip(ubicaciones, codificaciones):
            matches = face_recognition.compare_faces(rostros_conocidos, codificacion)
            nombre = "Desconocido"

            if True in matches:
                indice = matches.index(True)
                nombre = nombres_conocidos[indice]
            else:
                # 👇 Enviar alerta si pasó el tiempo de espera
                if time.time() - ultimo_envio > espera_alerta:
                    enviar_alerta(frame)
                    ultimo_envio = time.time()

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generar_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
