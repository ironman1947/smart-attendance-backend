import os
import requests

url = "http://127.0.0.1:8000/mark-attendance/"
folder = "captured_faces/252600024_Om"

# Just grab the very first image in the folder to act as our "classroom snapshot"
test_image = [f for f in os.listdir(folder) if f.endswith(".jpg")][0]
file_path = os.path.join(folder, test_image)

# Package it for the API
files = [("file", (test_image, open(file_path, "rb"), "image/jpeg"))]

print(f"Sending {test_image} to the attendance scanner...")

# Send the POST request
response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())