import cv2
import numpy as np
import face_recognition
import subprocess
import os
import time
from picamera2 import Picamera2

def speak(text):
    print(f"Speaking: {text}")
    subprocess.run(['espeak-ng', '-a', '200', '-g', '5', '-p', '50', '-s', '130', text])

known_encodings = []
known_names = []
known_faces_dir = "known_faces"
print("Loading known faces...")

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(filename)[0]
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)
            print(f"Loaded: {name}")
        else:
            print(f"No face found in {filename}, skipping")

print(f"Loaded {len(known_names)} known faces")

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()
print("Camera started")

last_spoken = {}
speak_cooldown = 5

try:
    while True:
        frame = picam2.capture_array()
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_names[best_match]

            current_time = time.time()
            last_time = last_spoken.get(name, 0)
            if current_time - last_time > speak_cooldown:
                if name == "Unknown":
                    speak("I see an unknown person")
                else:
                    speak(f"Hello {name}")
                last_spoken[name] = current_time
            print(f"Recognised: {name}")

except KeyboardInterrupt:
    print("Stopped")
finally:
    picam2.stop()
