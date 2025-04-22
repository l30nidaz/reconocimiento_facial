import face_recognition
import cv2
import pickle
import time
import os
from alertas import enviar_alerta  # 👈 Importamos la función de alerta

# Cargar datos entrenados
with open("rostros_entrenados.pkl", "rb") as f:
    datos_entrenados = pickle.load(f)

rostros_conocidos = datos_entrenados["rostros"]
nombres_conocidos = datos_entrenados["nombres"]

video = cv2.VideoCapture(0)

procesar_frame = 0
ultima_alerta = 0
intervalo_alerta = 30  # segundos

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Solo procesar cada 5 frames
    if procesar_frame % 5 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb_frame)
        codificaciones = face_recognition.face_encodings(rgb_frame, ubicaciones)

        for ubicacion, codificacion in zip(ubicaciones, codificaciones):
            resultados = face_recognition.compare_faces(rostros_conocidos, codificacion)
            nombre = "Desconocido"

            if True in resultados:
                indice = resultados.index(True)
                nombre = nombres_conocidos[indice]
            else:
                # 📸 Guardar imagen del rostro desconocido
                timestamp = int(time.time())
                ruta_imagen = f"desconocido_{timestamp}.jpg"
                cv2.imwrite(ruta_imagen, frame)

                # 📨 Enviar alerta si ha pasado suficiente tiempo desde la última
                if time.time() - ultima_alerta > intervalo_alerta:
                    enviar_alerta(ruta_imagen)
                    ultima_alerta = time.time()

            top, right, bottom, left = ubicacion
            color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    procesar_frame += 1

    cv2.imshow("Reconocimiento Facial", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
