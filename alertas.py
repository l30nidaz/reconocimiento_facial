# alertas.py
import smtplib
from email.message import EmailMessage
import cv2
import os
from datetime import datetime

def enviar_alerta(imagen_desconocida):
    # Guardar la imagen temporalmente
    nombre_archivo = f"desconocido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    ruta_imagen = os.path.join("alertas", nombre_archivo)
    os.makedirs("alertas", exist_ok=True)
    cv2.imwrite(ruta_imagen, imagen_desconocida)

    # Configura tu correo
    remitente = "jwlioabel@gmail.com"
    contraseña = "amphnjilayzhcqyj"
    destinatario = "jwlioabel@gmail.com"

    mensaje = EmailMessage()
    mensaje["Subject"] = "¡Alerta! Rostro desconocido detectado"
    mensaje["From"] = remitente
    mensaje["To"] = destinatario
    mensaje.set_content("Se ha detectado un rostro desconocido. Se adjunta la imagen.")

    with open(ruta_imagen, "rb") as img:
        mensaje.add_attachment(img.read(), maintype="image", subtype="jpeg", filename=nombre_archivo)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(remitente, contraseña)
        smtp.send_message(mensaje)
