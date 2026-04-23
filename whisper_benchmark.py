from faster_whisper import WhisperModel
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5

def record_audio(filename="test_audio.wav"):
    print(f"Recording for {DURATION} seconds... SPEAK NOW")
    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    wav.write(filename, SAMPLE_RATE, audio)
    print("Recording done!")

def benchmark_model(model_size, audio_file):
    print(f"\nTesting model: {model_size}")
    
    # Time how long loading takes
    start = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_time = time.time() - start
    
    # Time how long transcription takes
    start = time.time()
    segments, info = model.transcribe(audio_file, beam_size=5)
    result = " ".join([s.text for s in segments])
    transcribe_time = time.time() - start
    
    print(f"Result: {result}")
    print(f"Load time: {load_time:.2f}s")
    print(f"Transcribe time: {transcribe_time:.2f}s")
    return result, transcribe_time

# Main
print("=== Whisper Benchmark on Raspberry Pi ===")
record_audio()

for model_size in ["tiny", "base"]:
    benchmark_model(model_size, "test_audio.wav")

print("\nBenchmark complete!")
