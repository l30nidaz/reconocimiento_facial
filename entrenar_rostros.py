import face_recognition
import pickle
import os

# Cargar rostros entrenados existentes
if os.path.exists("rostros_entrenados.pkl"):
    with open("rostros_entrenados.pkl", "rb") as archivo:
        rostros_codificados, nombres = pickle.load(archivo)
else:
    rostros_codificados = []
    nombres = []

# Cargar la nueva imagen
nueva_imagen = face_recognition.load_image_file("rostros_conocidos/Doris Garcia.jpeg")
nuevos_codificados = face_recognition.face_encodings(nueva_imagen)

# Si se encuentran rostros, agregar a los datos
if nuevos_codificados:
    rostros_codificados.append(nuevos_codificados[0])
    nombres.append("Nuevo Nombre")  # El nombre del nuevo rostro

# Guardar los datos actualizados
with open("rostros_entrenados.pkl", "wb") as archivo:
    pickle.dump([rostros_codificados, nombres], archivo)

print("Nuevo rostro registrado correctamente.")
