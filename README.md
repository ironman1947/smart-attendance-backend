# 🎓 Smart Attendance System — AI Facial Recognition Backend

A high-accuracy, real-time facial recognition attendance backend built with **Python, FastAPI, and Deep Learning**.

This system acts as the core intelligence for a smart classroom application. It follows a **"Thin Client, Thick Server"** architecture designed to receive image frames from an Android application, extract **512-dimensional face embeddings** using state-of-the-art neural networks, and perform real-time verification against a secure SQLite vector database.

---

## ✨ Key Features

* **Zero-Training Registration**
  Extracts and stores mathematical face embeddings instantly without retraining the underlying AI model.

* **Batch Processing — Classroom Mode**
  Handles multiple faces detected in a single wide-angle frame and can mark multiple students as present simultaneously.

* **Privacy-First Storage**
  Raw images are processed in memory and immediately discarded. Only encrypted 512-dimensional binary BLOBs are persisted in the database.

* **Deduplication Ready**
  Designed to integrate with frontend `Set`-based logic to prevent duplicate attendance records during row-by-row scanning.

* **High Accuracy**
  Powered by **InsightFace / ArcFace** using the `buffalo_l` model for feature extraction and cosine similarity for face matching.

---

## 🛠️ Tech Stack

| Component          | Technology                |
| ------------------ | ------------------------- |
| Backend Framework  | FastAPI, Uvicorn          |
| Computer Vision    | OpenCV (`cv2`)            |
| Deep Learning      | InsightFace, ONNX Runtime |
| Model              | ArcFace / `buffalo_l`     |
| Database           | SQLite3                   |
| Vector Mathematics | NumPy                     |
| Execution          | CPU                       |

---

## 📁 Project Structure

```text
smart-attendance/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── captured_faces/
│   └── # Local storage for captured burst frames
│
├── client_tools/
│   └── auto_capture.py
│
├── attendance.db
├── live_scanner.py
├── test_attendance.py
├── test_upload.py
├── README.md
└── .gitignore
```

### File Description

| File / Directory               | Description                                                 |
| ------------------------------ | ----------------------------------------------------------- |
| `backend/main.py`              | Core FastAPI server and SQLite initialization               |
| `backend/requirements.txt`     | Python dependencies                                         |
| `captured_faces/`              | Temporary storage for captured face frames                  |
| `client_tools/auto_capture.py` | Webcam-based burst image capture utility                    |
| `attendance.db`                | SQLite database containing stored face embeddings           |
| `live_scanner.py`              | Real-time webcam testing and multi-face tracking            |
| `test_attendance.py`           | Automated test for the `/mark-attendance/` endpoint         |
| `test_upload.py`               | Automated test for the `/register/` endpoint                |
| `.gitignore`                   | Prevents sensitive and generated files from being committed |

> **Note:** Virtual environments, cache files, raw `.jpg` images, and sensitive data such as `attendance.db` are strictly excluded from version control using `.gitignore`.

---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/ironman1947/smart-attendance-backend.git
cd smart-attendance
```

---

## 2. Create a Virtual Environment

Creating a virtual environment is recommended to keep project dependencies isolated.

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

> **Note:** InsightFace will automatically download the `buffalo_l` model weights (approximately 300 MB) to `~/.insightface` during the first run.

---

## 4. Start the FastAPI Server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📡 API Endpoints

## 1. Register a Student

### Endpoint

```http
POST /register/
```

### Content Type

```text
multipart/form-data
```

### Parameters

| Parameter    | Type  | Description                                  |
| ------------ | ----- | -------------------------------------------- |
| `student_id` | Text  | Unique student PRN / ID                      |
| `files`      | Files | Multiple `.jpg` images of the student's face |

### Example

```text
student_id = 252600024_Om
files = [image1.jpg, image2.jpg, ..., image20.jpg]
```

The backend processes the uploaded images, extracts the face embeddings, calculates their mean representation, and stores the resulting **512-dimensional embedding** in the SQLite database.

### Response

```json
{
  "message": "Student registered successfully"
}
```

---

## 2. Mark Attendance

### Endpoint

```http
POST /mark-attendance/
```

### Content Type

```text
multipart/form-data
```

### Parameters

| Parameter | Type | Description                                 |
| --------- | ---- | ------------------------------------------- |
| `file`    | File | A single image containing one or more faces |

### Example Response

```json
{
  "faces_detected": 3,
  "present": [
    "252600024_Om",
    "102_Jay",
    "103_Varad"
  ]
}
```

The endpoint:

1. Receives an image frame.
2. Detects all faces present in the frame.
3. Generates a 512-dimensional embedding for each detected face.
4. Compares each embedding against registered students.
5. Uses cosine similarity for verification.
6. Returns the recognized student IDs.

---

# 🧠 Face Recognition Pipeline

```text
                Android Application
                        │
                        │ Image Frame
                        ▼
                ┌───────────────┐
                │   FastAPI     │
                │    Server     │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    InsightFace│
                │   Face Detect │
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    ArcFace    │
                │ 512-D Embedding│
                └───────┬───────┘
                        │
                        ▼
                ┌───────────────┐
                │    SQLite     │
                │ Vector Storage│
                └───────┬───────┘
                        │
                        ▼
                Cosine Similarity
                        │
                        ▼
                ┌───────────────┐
                │   Attendance  │
                │    Result     │
                └───────────────┘
```

---

# 🎯 Recognition Strategy

The system uses **face embeddings** rather than storing or comparing raw images.

Each detected face is converted into a numerical representation:

```text
Face Image
    ↓
Face Detection
    ↓
ArcFace Model
    ↓
512-Dimensional Embedding
    ↓
Cosine Similarity
    ↓
Identity Match
```

The current matching strategy uses a cosine similarity threshold of approximately:

```text
Threshold > 0.50
```

> The optimal threshold should be calibrated against the actual classroom dataset and environmental conditions before production deployment.

---

# 🧪 Local Testing

The repository includes several scripts for testing the complete pipeline without requiring the Android application or Postman.

## 1. Capture Face Data

Run:

```bash
python client_tools/auto_capture.py
```

This opens the webcam and captures approximately **20 frames** of a student's face.

The captured frames are stored temporarily in:

```text
captured_faces/
```

---

## 2. Test Student Registration

Run:

```bash
python test_upload.py
```

This uploads the captured face images to:

```text
/register/
```

The backend then generates and stores the student's face embedding.

---

## 3. Test Attendance

Run:

```bash
python test_attendance.py
```

This sends a single image frame to:

```text
/mark-attendance/
```

and displays the recognized student IDs returned by the API.

---

## 4. Live Webcam Tracking

Run:

```bash
python live_scanner.py
```

This starts a real-time webcam scanner that sends frames to the backend and displays recognition results and attendance status.

---

# 🔐 Privacy & Security

The system is designed around a **privacy-first biometric storage model**.

### Data Flow

```text
Raw Image
   │
   ▼
Processed in RAM
   │
   ▼
Face Embedding Generated
   │
   ▼
Raw Image Discarded
   │
   ▼
Embedding Stored
```

The intended architecture avoids permanently storing raw face photographs.

Instead, the system stores numerical face representations in the SQLite database.

> **Important:** A face embedding is still biometric data. In a production deployment, the database should be encrypted at rest, access-controlled, backed up securely, and handled according to applicable privacy and biometric-data laws.

---

# 📊 System Architecture

```text
┌──────────────────────────────┐
│       Android Client        │
│                              │
│  Camera → Image Frames      │
└───────────────┬──────────────┘
                │
                │ HTTP / Multipart
                ▼
┌──────────────────────────────┐
│          FastAPI             │
│                              │
│  /register/                  │
│  /mark-attendance/           │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│        InsightFace           │
│                              │
│  Face Detection              │
│  ArcFace Embeddings          │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│        Vector Matching       │
│                              │
│     Cosine Similarity        │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│           SQLite             │
│                              │
│  Student ID + Face Vector    │
└──────────────────────────────┘
```

---

# 🔄 Registration Workflow

```text
Student
   │
   ▼
Capture ~20 Face Images
   │
   ▼
Upload Images
   │
   ▼
Face Detection
   │
   ▼
Generate 512-D Embeddings
   │
   ▼
Calculate Mean Embedding
   │
   ▼
Store Embedding + Student ID
```

This approach provides **zero-training registration** because adding a new student does not require retraining the ArcFace model.

---

# 🏫 Classroom Attendance Workflow

```text
Classroom Camera
       │
       ▼
Wide-Angle Frame
       │
       ▼
Multiple Face Detection
       │
       ├──────────► Student 1 → Match
       │
       ├──────────► Student 2 → Match
       │
       ├──────────► Student 3 → Match
       │
       └──────────► Unknown → Ignore
                         │
                         ▼
                  Attendance Result
```

This enables multiple students to be recognized from a single classroom frame.

---

# ⚡ Performance Considerations

The current backend is configured for **CPU-based ONNX Runtime execution**.

For larger classrooms or higher frame rates, future versions can be optimized using:

* GPU acceleration
* Batch inference
* Embedding caching
* Vector indexes
* FAISS or another dedicated vector-search engine
* Frame skipping
* Multi-threaded processing
* Asynchronous inference pipelines



---

# 👥 Credits & Core Team

### Machine Learning Architecture & Backend Development

* **Om Pradip Chougule** — [@ironman1947](https://github.com/ironman1947)
* **Harsh Desai** — [@profharsh2026-prog](https://github.com/profharsh2026-prog)

---

# 📄 License

This project is currently intended for educational and development purposes.

