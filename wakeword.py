import pyaudio
import numpy as np
from openwakeword.model import Model
import subprocess
import time

# Try importing pidog
try:
    from pidog import Pidog
    dog = Pidog()
    dog_available = True
    print("PiDog connected")
except Exception as e:
    print(f"PiDog not connected: {e}")
    dog_available = False

def speak(text):
    print(f"Speaking: {text}")
    subprocess.run(['espeak-ng', '-a', '200', '-g', '5', '-p', '50', '-s', '130', text])

def wake_action():
    speak("Yes, I am here")

    if dog_available:
        try:
            # Stand up
            print("Standing up...")
            dog.do_action('stand', speed=50)
            time.sleep(1)

            # Wave
            print("Waving...")
            dog.do_action('wave_right', speed=50)
            time.sleep(2)

            # Back to stand
            dog.do_action('stand', speed=50)
            print("Done")

        except Exception as e:
            print(f"PiDog action error: {e}")

# Load wake word model
print("Loading wake word model...")
model = Model()
print("Model loaded")

# Setup mic
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=1280
)

print("Listening for 'Hey Jarvis'...")

# Cooldown to prevent multiple triggers
last_triggered = 0
cooldown = 10

try:
    while True:
        data = stream.read(1280, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        prediction = model.predict(audio_data)

        for key, value in prediction.items():
            if value > 0.5:
                current_time = time.time()
                if current_time - last_triggered > cooldown:
                    print(f"Wake word detected: {key}")
                    last_triggered = current_time

                    # Stop mic so speech doesn't re-trigger
                    stream.stop_stream()

                    wake_action()

                    # Flush leftover audio then restart mic
                    stream.start_stream()
                    for _ in range(20):
                        stream.read(1280, exception_on_overflow=False)

                    print("Listening again...")

except KeyboardInterrupt:
    print("Stopped")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    if dog_available:
        dog.close()
