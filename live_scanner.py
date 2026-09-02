import cv2
import requests
import time

url = "http://127.0.0.1:8000/mark-attendance/"
cap = cv2.VideoCapture(0)

last_check_time = time.time()
present_students = []

print("Starting live attendance scanner... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Send a frame to the backend every 2 seconds to prevent video lag
    if time.time() - last_check_time > 2.0:
        # Encode the frame in memory as a JPG
        _, buffer = cv2.imencode('.jpg', frame)
        files = [("file", ("live_frame.jpg", buffer.tobytes(), "image/jpeg"))]
        
        try:
            # Fire it to FastAPI
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                present_students = data.get("present", [])
        except Exception as e:
            print("API connection error:", e)

        last_check_time = time.time()

    # Create the visual UI on the live feed
    cv2.putText(frame, "Live Attendance Feed", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    y_offset = 80
    if present_students:
        for student in present_students:
            cv2.putText(frame, f"✅ Present: {student}", (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
    else:
        cv2.putText(frame, "Scanning for students...", (20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Live Attendance Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()