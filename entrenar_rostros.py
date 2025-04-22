import face_recognition
import os
import pickle
from PIL import Image

directorio_rostros = "rostros_conocidos"
rostros = []
nombres = []

for nombre_archivo in os.listdir(directorio_rostros):
    ruta = os.path.join(directorio_rostros, nombre_archivo)
    imagen = face_recognition.load_image_file(ruta)
    codificaciones = face_recognition.face_encodings(imagen)
    
    if codificaciones:
        rostros.append(codificaciones[0])
        nombre = os.path.splitext(nombre_archivo)[0]
        nombres.append(nombre)
    else:
        print(f"No se encontró un rostro en {nombre_archivo}")

# Guardar como diccionario
datos_entrenados = {
    "rostros": rostros,
    "nombres": nombres
}

with open("rostros_entrenados.pkl", "wb") as f:
    pickle.dump(datos_entrenados, f)

print("Entrenamiento completado y datos guardados en 'rostros_entrenados.pkl'")
