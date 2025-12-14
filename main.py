import cv2
import mediapipe as mp
import math
import os
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

st.set_page_config(page_title="Gestos IA", layout="centered")

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.cache_resource
def load_meme_images():
    img_paths = {
        "normal":    "memes/normal.png", 
        "happy":     "memes/happy.jpg", 
        "shock":     "memes/shock.jpg",
        "turn_left": "memes/left.jpg",
        "turn_right":"memes/right.png",
        "one_finger":"memes/monkey.jpg", 
        "look_up":   "memes/lookup.jpg"
    }
    loaded_images = {}
    print("--- Cargando imágenes en servidor ---")
    for key, path in img_paths.items():
        if os.path.exists(path):
            img = cv2.imread(path)
            loaded_images[key] = img
        else:
            loaded_images[key] = None
    return loaded_images

MEME_IMAGES = load_meme_images()

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)

class VideoProcessor(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        h, w, _ = img.shape
        if w > 640:
            scale = 640 / w
            img = cv2.resize(img, (640, int(h * scale)))
            h, w, _ = img.shape

        img.flags.writeable = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results_face = face_mesh.process(img_rgb)
        results_hands = hands_detector.process(img_rgb)
        
        img.flags.writeable = True

        # Variables por defecto
        current_meme = "normal"
        status_text = "NEUTRAL"
        color_text = (255, 255, 255)

        is_one_finger_up = False
        is_looking_up = False
        ratio_open = 0
        yaw_ratio = 1.0
        is_happy = False

        # Mano
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                index_tip_y = hand_landmarks.landmark[8].y
                middle_tip_y = hand_landmarks.landmark[12].y
                ring_tip_y = hand_landmarks.landmark[16].y
                pinky_tip_y = hand_landmarks.landmark[20].y
                
                middle_pip_y = hand_landmarks.landmark[10].y # Nudillo medio

                if (index_tip_y < middle_tip_y and 
                    index_tip_y < ring_tip_y and 
                    index_tip_y < pinky_tip_y and
                    middle_tip_y > middle_pip_y):
                    is_one_finger_up = True
                    break 
                
        # Cara
        if results_face.multi_face_landmarks:
            for face_landmarks in results_face.multi_face_landmarks:
                
                # Puntos clave
                up_lip = face_landmarks.landmark[13]
                low_lip = face_landmarks.landmark[14]
                left_mouth = face_landmarks.landmark[61]
                right_mouth = face_landmarks.landmark[291]
                nose_tip = face_landmarks.landmark[1]
                left_edge = face_landmarks.landmark[234]
                right_edge = face_landmarks.landmark[454]
                nose_bridge = face_landmarks.landmark[168]
                chin = face_landmarks.landmark[152]

                # Convertir a píxeles
                pt_ul = (int(up_lip.x * w), int(up_lip.y * h))
                pt_ll = (int(low_lip.x * w), int(low_lip.y * h))
                pt_lm = (int(left_mouth.x * w), int(left_mouth.y * h))
                pt_rm = (int(right_mouth.x * w), int(right_mouth.y * h))
                nose_x = int(nose_tip.x * w)
                left_edge_x = int(left_edge.x * w)
                right_edge_x = int(right_edge.x * w)

                # Boca
                mouth_open_dist = calculate_distance(pt_ul, pt_ll)
                mouth_width = calculate_distance(pt_lm, pt_rm)
                if mouth_width == 0: mouth_width = 1
                ratio_open = mouth_open_dist / mouth_width

                # Giro lateral
                dist_l = abs(nose_x - left_edge_x)
                dist_r = abs(right_edge_x - nose_x)
                if dist_r == 0: dist_r = 1
                yaw_ratio = dist_l / dist_r

                # Feliz
                if (left_mouth.y < nose_tip.y + 0.04) and (right_mouth.y < nose_tip.y + 0.04):
                    is_happy = True

                # Mirar Arriba
                face_vertical_height = calculate_distance(
                    (int(nose_bridge.x*w), int(nose_bridge.y*h)), 
                    (int(chin.x*w), int(chin.y*h))
                )
                if face_vertical_height == 0: face_vertical_height = 1

                nose_vertical_dist = (nose_tip.y * h) - (nose_bridge.y * h)
                pitch_ratio = nose_vertical_dist / face_vertical_height

                if pitch_ratio < 0.07: 
                    is_looking_up = True

        # Prioridades
        if is_one_finger_up:
            current_meme = "one_finger"
            status_text = "De hecho"
        
        elif is_looking_up:
            current_meme = "look_up"
            status_text = "Arriba"

        elif ratio_open > 0.35: 
            current_meme = "shock"
            status_text = "OH"

        elif yaw_ratio < 0.4:
            current_meme = "turn_left" 
            status_text = "Izq"
            
        elif yaw_ratio > 2.5:
            current_meme = "turn_right"
            status_text = "Der"

        elif is_happy:
            current_meme = "happy"
            status_text = ":)"

        cv2.putText(img, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 2)

        meme_raw = MEME_IMAGES.get(current_meme)
        if meme_raw is not None:
            meme_display = cv2.resize(meme_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            meme_display = np.zeros((h, w, 3), dtype="uint8")
            cv2.putText(meme_display, "FALTA IMAGEN", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        combined = np.vstack((img, meme_display))
        
        return av.VideoFrame.from_ndarray(combined, format="bgr24")

# Interfaz
st.title("Detector de Gestos 📱")
st.write("Dale permisos a la cámara. Si va lento, asegúrate de tener buena luz.")

webrtc_streamer(
    key="gestos",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)