import cv2
import os
import time
from insightface.app import FaceAnalysis
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from quality_check import check_image_quality

def capture_registration_burst(num_images=20, save_dir="captured_faces"):
    """Automatically captures a burst of images for student registration."""
    
    print("--- Student Registration Image Capture ---")
    name = input("Enter Student Name: ").strip()
    prn = input("Enter Student PRN: ").strip()
    
    if not name or not prn:
        print("Error: Name and PRN cannot be empty.")
        return

    folder_name = f"{prn}_{name.replace(' ', '_')}"
    student_path = os.path.join(save_dir, folder_name)
    os.makedirs(student_path, exist_ok=True)
    
    existing_images = [f for f in os.listdir(student_path) if f.endswith('.jpg')]
    start_index = len(existing_images)
    current_count = 0
    
    print("\nLoading AI face detector (InsightFace)...")
    # Initialize InsightFace detector
    face_app = FaceAnalysis(name='buffalo_l')
    # ctx_id=-1 forces CPU execution, which avoids Apple Silicon GPU compatibility issues
    face_app.prepare(ctx_id=-1, det_size=(640, 640)) 
    
    cap = cv2.VideoCapture(0)
    
    print(f"\nStarting capture for {name} ({prn}).")
    print(f"Data will be saved to: {student_path}")
    print("Please look at the camera... capturing will start in 2 seconds.")
    time.sleep(2) 
    
    capture_delay = 0.2 
    
    while current_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
            
        # Detect face using InsightFace
        faces = face_app.get(frame)
        
        display_frame = frame.copy()
        
        # Only capture if exactly one face is clearly visible
        if len(faces) == 1:
        # Extract coordinates for the bounding box
            box = faces[0].bbox.astype(int)
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # NEW: quality gate before this frame counts as a good capture
            quality_ok, issues = check_image_quality(frame, box)

            if quality_ok:
                file_number = start_index + current_count
                img_name = os.path.join(student_path, f"frame_{file_number:04d}.jpg")
                cv2.imwrite(img_name, frame)

                current_count += 1
                print(f"Captured {current_count}/{num_images}")

                time.sleep(capture_delay)
            else:
            # Bad frame -> reject it and tell the student what to fix.
            # The while loop keeps running, so this is effectively
            # "try that capture again" in real time.
                message = issues[0]
                cv2.putText(display_frame, message, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"Rejected frame: {message}") 
        else:
            cv2.putText(display_frame, "Ensure exactly ONE face is visible", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
        cv2.imshow("Registration Capture - Press 'q' to quit early", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Successfully saved {current_count} new images in '{folder_name}'.")
    print(f"📁 Total images now in this folder: {start_index + current_count}")

if __name__ == "__main__":
    capture_registration_burst()