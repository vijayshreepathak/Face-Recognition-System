import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pickle
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Facial Recognition System",
    page_icon="👤",
    layout="wide"
)

# Create necessary directories
os.makedirs("known_faces", exist_ok=True)
os.makedirs("encodings", exist_ok=True)
os.makedirs("models", exist_ok=True)

# DNN model paths
prototxt_path = os.path.join("models", "deploy.prototxt")
model_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")

# Correct URLs for model files
prototxt_url = "https://github.com/opencv/opencv/raw/4.x/samples/dnn/face_detector/deploy.prototxt"
caffemodel_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Download if not present
import urllib.request

def download_file(url, path, description):
    try:
        st.info(f"Downloading DNN face detector model ({description})...")
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        st.error(f"Failed to download {description}: {e}")
        raise

if not os.path.exists(prototxt_path):
    download_file(prototxt_url, prototxt_path, "prototxt")
if not os.path.exists(model_path):
    download_file(caffemodel_url, model_path, "caffemodel")

face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Helper: DNN face detection

def detect_faces_dnn(image_np, conf_threshold=0.7):
    h, w = image_np.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            faces.append((x1, y1, x2 - x1, y2 - y1))
    return faces

# Session state for known faces
if 'known_face_encodings' not in st.session_state:
    st.session_state.known_face_encodings = []
    st.session_state.known_face_names = []

def load_known_faces():
    try:
        with open('encodings/face_encodings.pkl', 'rb') as f:
            data = pickle.load(f)
            st.session_state.known_face_encodings = data['encodings']
            st.session_state.known_face_names = data['names']
    except FileNotFoundError:
        st.session_state.known_face_encodings = []
        st.session_state.known_face_names = []

def save_known_faces():
    data = {
        'encodings': st.session_state.known_face_encodings,
        'names': st.session_state.known_face_names
    }
    with open('encodings/face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)

load_known_faces()

# --- UI ---
st.markdown("""
<style>
    .big-font { font-size:40px !important; font-weight: bold; }
    .sub-font { font-size:20px !important; }
    .stTabs [data-baseweb="tab"] { font-size: 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-font" style="color:#6c63ff;">👤 Facial Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-font">This system allows you to:<ul><li>Register new faces (from webcam or file)</li><li>Recognize faces in images</li><li>View registered faces</li></ul></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📸 Register Face", "🔎 Recognize Faces", "🗂️ View Registered Faces"
])

with tab1:
    st.markdown("#### Register New Face")
    st.write("Upload a face image or use your webcam:")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], key="register")
    with col2:
        webcam_image = st.camera_input("Or capture from webcam")

    image = None
    if webcam_image is not None:
        image = Image.open(webcam_image)
        st.success("Webcam image captured!")
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("Image uploaded!")

    if image is not None:
        image_np = np.array(image)
        faces = detect_faces_dnn(image_np)
        if len(faces) == 0:
            st.error("No face detected in the image. Please try another image.")
        else:
            # Show all detected faces and let user pick one
            st.image(image, caption="Detected faces highlighted below:", use_column_width=True)
            for idx, (x, y, w, h) in enumerate(faces):
                face_crop = image_np[y:y+h, x:x+w]
                st.image(face_crop, caption=f"Face #{idx+1}", width=150)
            face_idx = 0
            if len(faces) > 1:
                face_idx = st.number_input(f"Multiple faces detected. Select which face to register (1-{len(faces)}):", min_value=1, max_value=len(faces), value=1) - 1
            person_name = st.text_input("Enter the person's name:")
            if st.button("Register Face") and person_name:
                (x, y, w, h) = faces[face_idx]
                face_roi = cv2.cvtColor(image_np[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"known_faces/{person_name}_{timestamp}.jpg"
                Image.fromarray(image_np[y:y+h, x:x+w]).save(image_path)
                st.session_state.known_face_encodings.append(face_roi)
                st.session_state.known_face_names.append(person_name)
                save_known_faces()
                st.success(f"Face registered successfully for {person_name}!")
                st.image(face_crop, caption=f"Registered face: {person_name}", width=200)

with tab2:
    st.markdown("#### Recognize Faces")
    uploaded_file = st.file_uploader("Upload an image to recognize faces", type=["jpg", "jpeg", "png"], key="recognize")
    webcam_image = st.camera_input("Or capture from webcam for recognition")
    image = None
    if webcam_image is not None:
        image = Image.open(webcam_image)
        st.success("Webcam image captured!")
    elif uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("Image uploaded!")
    if image is not None:
        image_np = np.array(image)
        faces = detect_faces_dnn(image_np)
        if len(faces) == 0:
            st.error("No faces detected in the image.")
        else:
            image_with_boxes = image_np.copy()
            for (x, y, w, h) in faces:
                face_roi = cv2.cvtColor(image_np[y:y+h, x:x+w], cv2.COLOR_RGB2GRAY)
                name = "Unknown"
                min_dist = float('inf')
                for i, known_face in enumerate(st.session_state.known_face_encodings):
                    try:
                        known_face_resized = cv2.resize(known_face, (w, h))
                        dist = np.sum((face_roi - known_face_resized) ** 2)
                        if dist < min_dist:
                            min_dist = dist
                            name = st.session_state.known_face_names[i]
                    except:
                        continue
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image_with_boxes, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            st.image(image_with_boxes, caption="Recognized Faces", use_column_width=True)

with tab3:
    st.markdown("#### Registered Faces")
    if not st.session_state.known_face_names:
        st.info("No faces registered yet.")
    else:
        cols = st.columns(3)
        for idx, name in enumerate(st.session_state.known_face_names):
            with cols[idx % 3]:
                st.markdown(f"**{name}**")
                face_images = [f for f in os.listdir("known_faces") if f.startswith(name)]
                if face_images:
                    latest_image = sorted(face_images)[-1]
                    st.image(f"known_faces/{latest_image}", width=200) 