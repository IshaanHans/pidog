import pyaudio
import numpy as np
from openwakeword.model import Model
import time

# Use onnx instead of tflite — tflite doesn't support Python 3.13
model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

CHUNK = 1280
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Listening for wake word...")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        predictions = model.predict(audio_np)
        for wake_word, score in predictions.items():
            if score > 0.5:
                print(f"Wake word detected! ({wake_word}: {score:.2f})")
                time.sleep(1)

except KeyboardInterrupt:
    print("Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
