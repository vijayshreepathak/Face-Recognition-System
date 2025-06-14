# Facial Recognition System

A modern, user-friendly facial recognition system built with Streamlit and OpenCV. This project allows you to register faces (from webcam or image), recognize faces in real time or from images, and view all registered faces. It uses a robust deep learning-based face detector for high accuracy.

## Features

- 📸 **Register new faces** from webcam or image upload
- 🔎 **Recognize faces** in real time (webcam) or from uploaded images
- 🗂️ **View all registered faces** in a gallery
- ⚡ **Robust face detection** using OpenCV DNN (deep learning)
- 🎨 **Modern, attractive UI** with Streamlit
- 🗃️ **Persistent storage** of registered faces and encodings

## Demo

![Demo Screenshot](demo_screenshot.png)

## Installation

1. **Clone this repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   > Make sure you have Python 3.7+

3. **First run will download the DNN face detector models automatically** (requires internet connection).

## Usage

Start the app with:
```bash
streamlit run app.py
```

Open your browser to the URL shown in the terminal (usually http://localhost:8501).

## How It Works

### Register Face
- Upload a photo or use your webcam to capture your face.
- The system detects all faces in the image. If there are multiple, you can select which one to register.
- Enter a name and register the face.

### Recognize Faces
- Upload a group photo or use your webcam.
- The system detects and recognizes all faces, labeling known faces and marking unknown ones.

### View Registered Faces
- See all faces you have registered, with names and thumbnails.

## Model Files
- The app will automatically download the required DNN model files (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`) into a `models/` directory on first run.

## Project Structure
```
Facial_Expression/
├── app.py
├── requirements.txt
├── README.md
├── models/                  # DNN model files (auto-downloaded)
├── known_faces/             # Registered face images
├── encodings/               # Face encodings (pickled)
```

## Requirements
- Python 3.7+
- streamlit
- opencv-contrib-python
- numpy
- pillow

## Tips
- For best results, use clear, well-lit images.
- Register multiple angles of the same person for better recognition.
- All data is stored locally; no images are uploaded to the cloud.

## License
MIT License

---

**Enjoy your modern facial recognition system!** 