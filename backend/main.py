import sqlite3
import numpy as np
import cv2
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from insightface.app import FaceAnalysis
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from quality_check import check_image_quality

# Initialize FastAPI application
app = FastAPI(title="Smart Attendance API")

# Initialize InsightFace model on CPU (ctx_id=-1)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

# Initialize SQLite Database
def init_db():
    with sqlite3.connect("attendance.db") as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                embedding BLOB
            )
        ''')

init_db()

def decode_image(file_bytes: bytes):
    """Decode raw image bytes to an OpenCV BGR numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.post("/register/")
async def register(
    student_id: str = Form(...), 
    files: List[UploadFile] = File(...)
):
    embeddings = []
    
    for file in files:
        img_bytes = await file.read()
        img = decode_image(img_bytes)
        
        if img is None:
            continue
            
        faces = face_app.get(img)
        
        if len(faces) == 1:
            box = faces[0].bbox.astype(int)
            quality_ok, _ = check_image_quality(img, box)
            if quality_ok:
                embeddings.append(faces[0].normed_embedding)
            
    if not embeddings:
        raise HTTPException(
            status_code=400, 
            detail="No clear single face detected across the uploaded images."
        )
        
    # Calculate the normalized mean vector
    mean_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    mean_embedding = mean_embedding.astype(np.float32)
    
    # Store only the binary embedding
    with sqlite3.connect("attendance.db") as conn:
        conn.execute(
            "REPLACE INTO students (student_id, embedding) VALUES (?, ?)", 
            (student_id, mean_embedding.tobytes())
        )
        
    return {
        "status": "success",
        "message": f"Student {student_id} registered successfully",
        "vectors_processed": len(embeddings)
    }

@app.post("/mark-attendance/")
async def mark_attendance(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = decode_image(img_bytes)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    faces = face_app.get(img)
    present_students = []
    
    with sqlite3.connect("attendance.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT student_id, embedding FROM students")
        db_records = cursor.fetchall()
        
    if not db_records:
        return {"faces_detected": len(faces), "present": []}
    
    for face in faces:
        test_emb = face.normed_embedding.astype(np.float32)
        best_match = None
        highest_sim = 0.0
        
        for student_id, emb_blob in db_records:
            saved_emb = np.frombuffer(emb_blob, dtype=np.float32)
            
            # Cosine similarity for normalized vectors
            sim = float(np.dot(test_emb, saved_emb) / (np.linalg.norm(test_emb) * np.linalg.norm(saved_emb)))
            
            if sim > highest_sim:
                highest_sim = sim
                best_match = student_id
                
        # Confidence threshold for Buffalo_L / ArcFace
        if highest_sim > 0.50 and best_match:
            present_students.append(best_match)
            
    return {
        "faces_detected": len(faces),
        "present": list(set(present_students))
    }