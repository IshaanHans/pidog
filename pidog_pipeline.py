from faster_whisper import WhisperModel

import sounddevice as sd

import scipy.io.wavfile as wav

import numpy as np

import os

from groq import Groq

SAMPLE_RATE = 16000

DURATION = 5

# Put your Groq API key here

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def record_audio(filename="command.wav"):

    print(f"🎤 Recording for {DURATION} seconds... SPEAK NOW")

    audio = sd.rec(

        int(DURATION * SAMPLE_RATE),

        samplerate=SAMPLE_RATE,

        channels=1,

        dtype='int16'

    )

    sd.wait()

    wav.write(filename, SAMPLE_RATE, audio)

    print("✅ Recording done!")

def transcribe(filename="command.wav"):

    print("💭 Transcribing...")

    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    segments, _ = model.transcribe(filename, beam_size=5)

    text = " ".join([s.text for s in segments])

    print(f"👤 You said: {text}")

    return text

def ask_groq(text):

    print("🤖 Asking AI...")

    response = client.chat.completions.create(

        model="llama-3.3-70b-versatile",

        messages=[

            {"role": "system", "content": """You are PiDog, a friendly 

            robotic dog companion. Keep responses to 1-2 sentences. 

            Be enthusiastic and dog-like."""},

            {"role": "user", "content": text}

        ]

    )

    reply = response.choices[0].message.content

    print(f"🐕 PiDog: {reply}")

    return reply

# Main loop

print("=== PiDog Voice Pipeline ===")

while True:

    input("\nPress ENTER to speak to PiDog...")

    record_audio()

    text = transcribe()

    if text.strip():

        ask_groq(text)

    else:

        print("❌ Couldn't hear anything, try again")
