import cv2
import mediapipe as mp
import math
import os
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

meme_images = {}

def load_images():
    img_paths = {
        "normal":    "memes/normal.png", 
        "happy":     "memes/happy.jpg", 
        "shock":     "memes/shock.jpg",
        "turn_left": "memes/left.jpg",
        "turn_right":"memes/right.png",
        "one_finger":"memes/monkey.jpg",
        "look_up":   "memes/lookup.jpg"
    }
        
    print("--- Cargando imágenes ---")
    for key, path in img_paths.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is None:
                print(f"[ERROR] No se pudo leer el archivo: {path}")
                meme_images[key] = None
            else:
                print(f"[OK] Cargada: {path}")
                meme_images[key] = img
        else:
            print(f"[AVISO] No encontrada: {path} (Se usará pantalla negra)")
            meme_images[key] = None
    print("-------------------------")

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)

def main():
    load_images()
    
    # Cambiar entre 0 y 1 si da error
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se detecta la cámara.")
        return

    print("Cámara iniciada. Pulsa 'q' para salir.")

    while True:
        success, image = cap.read()
        if not success:
            continue

        h, w, _ = image.shape

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results_face = face_mesh.process(image_rgb)
        results_hands = hands_detector.process(image_rgb) # Procesamos manos
        
        image.flags.writeable = True

        current_meme = "normal"
        status_text = "NEUTRAL"
        color_text = (255, 255, 255)

        is_one_finger_up = False
        is_looking_up = False
        ratio_open = 0
        yaw_ratio = 1.0
        is_happy = False

        # Para detectar las manos y el índice
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
      
                index_tip_y = hand_landmarks.landmark[8].y
                middle_tip_y = hand_landmarks.landmark[12].y
                ring_tip_y = hand_landmarks.landmark[16].y
                pinky_tip_y = hand_landmarks.landmark[20].y
                
                middle_pip_y = hand_landmarks.landmark[10].y

                if (index_tip_y < middle_tip_y and 
                    index_tip_y < ring_tip_y and 
                    index_tip_y < pinky_tip_y and
                    middle_tip_y > middle_pip_y):
                    is_one_finger_up = True
                    break 


        # Detección facial
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                
                up_lip = face_landmarks.landmark[13]
                low_lip = face_landmarks.landmark[14]
                left_mouth = face_landmarks.landmark[61]
                right_mouth = face_landmarks.landmark[291]
                nose_tip = face_landmarks.landmark[1]
                left_edge = face_landmarks.landmark[234]
                right_edge = face_landmarks.landmark[454]

                nose_bridge = face_landmarks.landmark[168]
                chin = face_landmarks.landmark[152]

                # Conversión a píxeles
                pt_ul = (int(up_lip.x * w), int(up_lip.y * h))
                pt_ll = (int(low_lip.x * w), int(low_lip.y * h))
                pt_lm = (int(left_mouth.x * w), int(left_mouth.y * h))
                pt_rm = (int(right_mouth.x * w), int(right_mouth.y * h))
                nose_x = int(nose_tip.x * w)
                left_edge_x = int(left_edge.x * w)
                right_edge_x = int(right_edge.x * w)

                # 1. Boca
                mouth_open_dist = calculate_distance(pt_ul, pt_ll)
                mouth_width = calculate_distance(pt_lm, pt_rm)
                if mouth_width == 0: mouth_width = 1
                ratio_open = mouth_open_dist / mouth_width

                # 2. Giro lateral
                dist_nose_to_left = abs(nose_x - left_edge_x)
                dist_nose_to_right = abs(right_edge_x - nose_x)
                if dist_nose_to_right == 0: dist_nose_to_right = 1
                yaw_ratio = dist_nose_to_left / dist_nose_to_right

                # 3. Feliz
                if (left_mouth.y < nose_tip.y + 0.04) and (right_mouth.y < nose_tip.y + 0.04):
                    is_happy = True

                face_vertical_height = calculate_distance(
                    (int(nose_bridge.x*w), int(nose_bridge.y*h)), 
                    (int(chin.x*w), int(chin.y*h))
                )
                if face_vertical_height == 0: face_vertical_height = 1

                nose_bridge_y = nose_bridge.y * h
                nose_tip_y = nose_tip.y * h
                nose_vertical_dist = nose_tip_y - nose_bridge_y
                
                pitch_ratio = nose_vertical_dist / face_vertical_height

                if pitch_ratio < 0.07: 
                    is_looking_up = True

        # Prioridad 1: Gesto de mano
        if is_one_finger_up:
            current_meme = "one_finger"
            status_text = "De hecho"
            
        # Prioridad 2: Mirar Arriba
        elif is_looking_up:
            current_meme = "look_up"
            status_text = "Arriba"

        # Prioridad 3: Sorpresa
        elif ratio_open > 0.35: 
            current_meme = "shock"
            status_text = "OH"

        # Prioridad 4: Giros laterales
        elif yaw_ratio < 0.4:
            current_meme = "turn_left" 
            status_text = "Izq"
            
        elif yaw_ratio > 2.5:
            current_meme = "turn_right"
            status_text = "Der"

        # Prioridad 5: Feliz
        elif is_happy:
            current_meme = "happy"
            status_text = ":)"

        cv2.putText(image, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2)

        meme_raw = meme_images.get(current_meme)
        if meme_raw is not None:
            meme_display = cv2.resize(meme_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            meme_display = np.zeros((h, w, 3), dtype="uint8")
            cv2.putText(meme_display, f"FALTA: {current_meme}", (20, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        combined_view = np.hstack((image, meme_display))
        cv2.imshow('Mi Proyecto IA - Gestos Avanzados', combined_view)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hands_detector.close()
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()