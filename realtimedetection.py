import cv2
import numpy as np
from keras.models import load_model

# Charger le modèle
model = load_model('emotiondetector.h5')  # Vérifie que ce fichier est bien dans le bon dossier

# Charger le classificateur Haar pour la détection des visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Dictionnaire des émotions
labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}


# Fonction pour préparer l'image
def extract_features(image):
    image = cv2.resize(image, (48, 48))  # Redimensionnement
    image = np.array(image, dtype=np.float32) / 255.0  # Normalisation
    image = image.reshape(1, 48, 48, 1)  # Reshape pour le modèle
    return image


# Vérifier si la webcam est accessible
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("❌ Erreur : Impossible d'ouvrir la webcam")
    exit()

while True:
    # Lire l'image depuis la webcam
    success, frame = webcam.read()
    if not success:
        print("❌ Erreur : Impossible de lire l'image de la webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]  # Extraire le visage
        features = extract_features(face)  # Prétraitement

        pred = model.predict(features)  # Prédiction
        emotion_label = labels[np.argmax(pred)]  # Émotion prédite

        # Dessiner un rectangle autour du visage et afficher l'émotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Affichage avec OpenCV
    cv2.imshow("Reconnaissance des émotions", frame)

    # Quitter si l'utilisateur appuie sur 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Libérer les ressources
webcam.release()
cv2.destroyAllWindows()

