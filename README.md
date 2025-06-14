# 👤 Facial Recognition System

A modern, user-friendly facial recognition system built with **Streamlit** and **OpenCV**. This application enables users to register faces (via webcam or image), perform real-time or image-based face recognition, and view all stored faces in a sleek, interactive UI.

## ✨ Features

- 📸 **Register Faces** — Capture via webcam or upload an image
- 🔍 **Real-Time Recognition** — Identify faces live through your webcam
- 🖼️ **Image Recognition** — Detect and label faces in uploaded group photos
- 🗂️ **Gallery View** — Browse through all registered users with name tags
- 🧠 **Deep Learning Face Detection** — Powered by OpenCV's DNN module
- 💾 **Persistent Local Storage** — Save face encodings and images
- 🎨 **Clean Streamlit UI** — Simple, responsive, and intuitive interface

---

## 🧪 Demo

## 🚀 Live Application

🔗 **Try it out here:**  
👉 [https://face-recognition-system-u0nj.onrender.com/](https://face-recognition-system-u0nj.onrender.com/)

> ⏳ *Note: App may take up to 60 seconds to load initially due to Render's free-tier cold start.*

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Facial_Recognition_System.git
cd Facial_Recognition_System

# 2. Install dependencies
pip install -r requirements.txt
````

> ✅ Make sure you have **Python 3.7+** installed.

> 🌐 On the first run, the app will **automatically download** the required DNN face detection model files.

---

## ▶️ Usage

To start the application locally:

```bash
streamlit run app.py
```

Then open your browser and go to `http://localhost:8501`.

---

## 🧠 How It Works

### 📝 Register Face

* Upload a photo or use your webcam to capture a face.
* The system detects all visible faces.
* You can select which face to register and assign a name.

### 🕵️ Recognize Faces

* Upload an image or enable webcam.
* The system identifies known individuals and highlights unknown faces.

### 🖼️ View Registered Faces

* Browse through all stored faces along with their assigned names.
* Helpful to confirm registrations or delete unwanted ones (if feature enabled).

---

## 📁 Project Structure

```
Facial_Recognition_System/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── models/                # Auto-downloaded DNN model files
├── known_faces/           # Stored face images
├── encodings/             # Pickled face encodings
```

---

## 🧾 Requirements

* Python 3.7+
* streamlit
* opencv-contrib-python
* numpy
* pillow

Install them using:

```bash
pip install -r requirements.txt
```

---

## 💡 Tips for Best Results

* Use **well-lit**, **front-facing** images for better accuracy.
* Register **multiple angles** of the same person.
* All data is stored **locally** – no cloud uploads or third-party servers involved.

---

## 📦 Model Files

No need for manual download! On the first run, the following model files are auto-downloaded into the `models/` folder:

* `deploy.prototxt`
* `res10_300x300_ssd_iter_140000.caffemodel`

These power the OpenCV DNN face detector for high-accuracy recognition.

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

> **Enjoy building and experimenting with your own facial recognition system!**
> Contributions and feedback are welcome 🤝

```

---
