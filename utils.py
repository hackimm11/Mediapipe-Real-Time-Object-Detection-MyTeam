import cv2
import numpy as np
from mediapipe.tasks.python import vision
import mediapipe as mp
import time


def print_detection(image: np.ndarray, detection_result: vision.ObjectDetectorResult) -> np.ndarray:
    """
    Annotate l'image en dessinant des rectangles et en affichant le nom de la catégorie 
    et le score de chaque détection.
    
    Args:
        image: Image RGB (numpy.ndarray).
        detection_result: Résultat de détection contenant les objets détectés.
    
    Returns:
        Image annotée.
    """
    for detection in detection_result.detections:
        # Récupérer le bounding box et calculer les coordonnées
        bbox = detection.bounding_box
        x_min = bbox.origin_x
        y_min = bbox.origin_y
        x_max = bbox.origin_x + bbox.width
        y_max = bbox.origin_y + bbox.height

        # Dessiner le rectangle sur l'image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255,0,0), 2)

        # Récupérer l'étiquette et le score
        category = detection.categories[0]
        label = f"{category.category_name} ({round(category.score, 2)})"
        # Placer le texte au-dessus du rectangle
        cv2.putText(image, label, (x_min, y_min - 12),
                    cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,0), 2)
    return image


def convert_to_mp_image(frame):
    """
    Convertit une frame OpenCV (BGR) en objet mp.Image (RGB) en appliquant l'effet miroir.
    
    Args:
        frame: Frame OpenCV (format BGR).
    
    Returns:
        mp.Image au format RGB.
    """
    # Appliquer l'effet miroir (selfie view)
    flipped = cv2.flip(frame, 1)
    # Convertir BGR en RGB
    rgb_frame = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

def get_timestamp():
    """
    Retourne le timestamp actuel en millisecondes.
    
    Returns:
        Timestamp en millisecondes (int).
    """
    return int(time.time() * 1000)

def setup_camera(camera_id=0, width=1280, height=720):
    """
    Initialise et configure la webcam.
    
    Args:
        camera_id: ID de la caméra.
        width: Largeur souhaitée de la capture.
        height: Hauteur souhaitée de la capture.
    
    Returns:
        Objet cv2.VideoCapture configuré.
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError("ERROR: Unable to open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def detection_callback(result, output_image, timestamp_ms, detection_results):
    """
    Fonction de callback pour traiter les résultats de détection en mode asynchrone.
    
    Args:
        result: Résultat de détection (vision.ObjectDetectorResult).
        output_image: mp.Image (peut être ignoré).
        timestamp_ms: Timestamp de la frame.
        detection_results: Liste dans laquelle ajouter le résultat.
    """
    result.timestamp_ms = timestamp_ms
    detection_results.append(result)



def draw_instructions(image, instruction="press \'q\' to exit or \'s\' to save snapshot ", position=(10, 20),
                      font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(0,255,0), thickness=2):
    """
    Dessine un texte d'instruction sur l'image.
    
    Returns:
        L'image annotée.
    """
    x, y = position
    if y is None:
        y = image.shape[0] - 10
    cv2.putText(image, instruction, (x, y), font, font_scale, color, thickness)
    return image