import argparse
import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import (print_detection,
    convert_to_mp_image, 
    get_timestamp, 
    setup_camera, 
    detection_callback,
    draw_instructions
)

def run(model: str, running_mode: str) -> None:
    """
    Exécute la détection d'objets sur le flux de la webcam en mode synchrone ou asynchrone.
    
    Args:
        model: Chemin du modèle TFLite de détection d'objets.
        running_mode: 'image' pour le mode synchrone ou 'live_stream' pour le mode asynchrone.
    """
    cap = setup_camera()  #  configurer et initialiser la caméra
    base_options = python.BaseOptions(model_asset_path=model) # modele de detction d'objet de mediapipe comme option pour le detcteur

    if running_mode == "image":
        # Mode synchrone (IMAGE)
        detector_options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            score_threshold=0.5,
            max_results=5
        )
        detector = vision.ObjectDetector.create_from_options(detector_options)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                sys.exit("ERROR: Unable to read from the webcam.")

            # Conversion et effet miroir sont gérés dans convert_to_mp_image
            mp_image = convert_to_mp_image(frame)
            detection_result = detector.detect(mp_image)

            output_image = mp_image.numpy_view()
            # Reconvertir en BGR pour cv2.imshow
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            if detection_result.detections:
                annotated_image = print_detection(output_image, detection_result)
            else:
                annotated_image = output_image.copy()

            # Draw instruction text using our utility function
            annotated_image = draw_instructions(annotated_image)

            cv2.imshow("Object Detector", annotated_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Sauvegarder la frame actuelle dans un fichier
                cv2.imwrite("results/snapshot.jpg",annotated_image)
                print("Capture enregistrée sous snapshot.jpg")
        detector.close()

    elif running_mode == "live_stream":
        # Mode asynchrone (LIVE_STREAM)
        detection_results = []
        detector_options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            score_threshold=0.5,
            max_results=5,
            result_callback=lambda result, img, ts: detection_callback(result, img, ts, detection_results)
        )
        detector = vision.ObjectDetector.create_from_options(detector_options)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                sys.exit("ERROR: Unable to read from the webcam.")

            mp_image = convert_to_mp_image(frame)
            detector.detect_async(mp_image, get_timestamp())

            output_image = mp_image.numpy_view()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

            if detection_result.detections:
                annotated_image = print_detection(output_image, detection_result)
                detection_results.clear()
            else:
                annotated_image = output_image.copy()

            # Draw instruction text using our utility function
            annotated_image = draw_instructions(annotated_image)

            cv2.imshow("Object Detector", annotated_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Sauvegarder la frame actuelle dans un fichier
                cv2.imwrite("results/snapshot.jpg",annotated_image)
                print("Capture enregistrée sous snapshot.jpg")
        detector.close()

    else:
        sys.exit("ERROR: Unsupported mode. Use 'image' or 'live_stream'.")

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help="Chemin du modèle TFLite de détection d'objets.",
        default='models/efficientdet_lite0_float16.tflite') # donne le meilleure rapport en terme de rapidité_inférence/précision
    parser.add_argument(
        '--mode',
        help="Mode d'exécution : 'image' (synchrone) ou 'live_stream' (asynchrone).",
        default='image') # personnelement je recommande ça, car le mode live_stream génre un retard dans le cas d'un hardware limité
    args = parser.parse_args()

    run(args.model, args.mode)

if __name__ == '__main__':
    main()
