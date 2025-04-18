from flask import Flask, render_template, Response
import cv2
import face_recognition
import pickle

app = Flask(__name__)

# Cargar datos entrenados
with open("rostros_entrenados.pkl", "rb") as f:
    datos_entrenados = pickle.load(f)

rostros_conocidos = datos_entrenados[0]
nombres_conocidos = datos_entrenados[1]

video = cv2.VideoCapture(0)

def generar_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ubicaciones_rostros = face_recognition.face_locations(rgb_frame)
        codificaciones_rostros = face_recognition.face_encodings(rgb_frame, ubicaciones_rostros)

        for (top, right, bottom, left), codificacion in zip(ubicaciones_rostros, codificaciones_rostros):
            coincidencias = face_recognition.compare_faces(rostros_conocidos, codificacion)
            nombre = "Desconocido"

            if True in coincidencias:
                indice = coincidencias.index(True)
                nombre = nombres_conocidos[indice]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
