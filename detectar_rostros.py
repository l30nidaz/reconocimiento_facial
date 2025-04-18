import cv2
import face_recognition
import pickle

# Cargar datos entrenados
with open("rostros_entrenados.pkl", "rb") as archivo:
    rostros_codificados, nombres = pickle.load(archivo)

# Iniciar cámara
video_captura = cv2.VideoCapture(0)

print("Presiona 'q' para salir...")

frame_count = 0
face_locations = []
face_names = []

while True:
    ret, frame = video_captura.read()
    if not ret:
        break

    # Redimensionar para más velocidad
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Solo procesar cada 5 frames
    if frame_count % 5 == 0:
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(rostros_codificados, encoding)
            nombre = "Desconocido"
            if True in matches:
                index = matches.index(True)
                nombre = nombres[index]
            face_names.append(nombre)

    # Mostrar resultados
    for (top, right, bottom, left), nombre in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, nombre, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Reconocimiento facial', frame)

    # Salir si se presiona 'q' o se cierra la ventana
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Reconocimiento facial', cv2.WND_PROP_VISIBLE) < 1:
        break

    frame_count += 1

video_captura.release()
cv2.destroyAllWindows()
