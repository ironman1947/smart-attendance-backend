import os
import requests

url = "http://127.0.0.1:8000/register/"
folder = "captured_faces/252600024_Om" # Update this if your folder name is different

# Gather all 20 images automatically
files = []
for filename in os.listdir(folder):
    if filename.endswith(".jpg"):
        file_path = os.path.join(folder, filename)
        files.append(("files", (filename, open(file_path, "rb"), "image/jpeg")))

data = {"student_id": "252600024_Om"}

print(f"Preparing to upload {len(files)} images...")

# Send the POST request to FastAPI
response = requests.post(url, data=data, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())