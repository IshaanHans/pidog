# 🐶 J.A.R.V.I.S — Just A Rather Very Intelligent Sniffer

**PiDog AI Companion | La Trobe University | Team PiDog 1 | 2026**

An AI-powered robotic dog built on the [SunFounder PiDog](https://github.com/sunfounder/pidog) platform. J.A.R.V.I.S extends the base hardware with multimodal AI — sign language translation, voice recognition, computer vision, and an adaptive personality system.

---

## Table of Contents

- [Overview](#overview)
- [Team](#team)
- [Hardware Setup Guide](#hardware-setup-guide)
- [Software Installation](#software-installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Known Issues & Sprint Notes](#known-issues--sprint-notes)
- [Demo](#demo)
- [Base Repository](#base-repository)

---

## Overview

J.A.R.V.I.S is not a pre-programmed robot — it learns, responds, and interacts. Its headline capability is **real-time sign language translation**: the onboard camera detects hand gestures using MediaPipe Hands, classifies them with a trained model, and speaks the meaning aloud via onboard TTS — acting as a live accessibility bridge between sign language users and those who don't know it.

---

## Team

| Name | Student ID | Role |
|---|---|---|
| Ishaan Hans | 21752626 | Project Lead & AI/NLP Engineer |
| Suhansa Benthotage | 21966347 | Integration & Demo Engineer |
| Luke Haykal | 21606899 | Hardware & Locomotion Engineer |
| Parth Jadav | 21372668 | Computer Vision Engineer |
| Dhil Balasingam | 21914856 | Documentation & QA Lead |

---

## Hardware Setup Guide

### What's in the kit

- SunFounder PiDog chassis, servo motors, and linkage arms
- Raspberry Pi 4 (4GB)
- Robot HAT with speaker output
- Pi Camera Module
- Ultrasonic sensor module
- Dual touch sensor
- Sound direction sensor
- RGB LED strip

### Step-by-step assembly

> ⚠️ **Read the known issues section below before starting assembly.** Two critical steps (SSH configuration and servo zeroing) must be done before you put anything together.

**1. Flash the SD card (do this first)**

Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to flash Raspberry Pi OS (64-bit, Lite recommended for Pi performance).

Before writing, open **Advanced Options** (`Ctrl+Shift+X`) and configure:
- ✅ Enable SSH
- ✅ Set username and password
- ✅ Configure WiFi SSID and password
- ✅ Set hostname (e.g. `jarvis.local`)

> See [Issue #1](#issue-1--ssh-disabled-by-default-on-raspberry-pi-os) — SSH is disabled by default and must be enabled here before first boot.

**2. Boot the Pi and verify SSH access**

```bash
# From your laptop on the same WiFi network:
ssh pi@jarvis.local
# or use the Pi's IP address if mDNS isn't working:
ssh pi@<PI_IP_ADDRESS>
```

To find the Pi's IP if hostname doesn't resolve:
```bash
# On your router's admin page, or:
nmap -sn 192.168.1.0/24 | grep -i raspberry
```

**3. Zero all servos before physical assembly**

> ⚠️ **Critical — do not attach servo horns or leg linkages until this step is complete.**  
> See [Issue #2](#issue-2--servo-misalignment-due-to-skipped-zero-calibration) for what happens if you skip this.

Install the SunFounder SDK first (see [Software Installation](#software-installation)), then run:

```bash
cd ~/pidog
python3 -c "
from pidog import Pidog
d = Pidog()
d.legs_move_raw([0]*16)
print('All servos at zero. Now attach the servo horns.')
input('Press Enter when done...')
d.close()
"
```

All servo shafts will rotate to their 0° neutral position. **Only then** attach the plastic servo horns and leg linkage arms.

**4. Assemble the chassis**

Follow the [SunFounder PiDog assembly guide](https://docs.sunfounder.com/projects/pidog/en/latest/). Key tips:

- Torque servo horn screws firmly — loose horns cause leg wobble
- Route cables away from joint pivot points
- Attach the Robot HAT before fitting the Pi into the chassis

**5. Verify locomotion**

```bash
cd ~/pidog
python3 basic_examples/stand.py
```

The dog should stand level with all four feet flat. If any leg is visibly angled or the chassis tilts, revisit servo zero calibration for that joint.

---

## Software Installation

```bash
# On the Raspberry Pi:
git clone https://github.com/IshaanHans/pidog.git
cd pidog

# Install SunFounder PiDog SDK
pip3 install -e .

# Install AI pipeline dependencies
pip3 install mediapipe opencv-python-headless numpy scikit-learn pyttsx3 anthropic

# Install audio dependencies
sudo apt-get install -y espeak-ng portaudio19-dev libatlas-base-dev
pip3 install pyaudio

# Optional: Porcupine wake word engine
pip3 install pvporcupine
# Get free API key at: https://console.picovoice.ai/
export PORCUPINE_ACCESS_KEY=your_key_here

# Optional: Claude API for sentence formation
export ANTHROPIC_API_KEY=your_key_here
```

**Run the sign language pipeline:**

```bash
# Always-on mode (best for demos)
python3 pipeline/main.py --always-on --llm

# Full mode with wake word ("Hey JARVIS")
python3 pipeline/main.py --llm --ppn hey-jarvis_raspberry-pi.ppn
```

---

## Project Structure

```
pidog/
├── pidog/              # SunFounder SDK (base hardware library)
├── basic_examples/     # Simple locomotion demos from SunFounder
├── examples/           # Extended behaviour examples
├── pipeline/           # J.A.R.V.I.S AI pipeline (our code)
│   ├── main.py         # Entry point — full live pipeline
│   ├── detector.py     # MediaPipe hand landmark extraction
│   ├── classifier.py   # Sign language classifier with smoothing
│   ├── wake_word.py    # Wake word detection (Porcupine / Vosk)
│   ├── tts.py          # Non-blocking text-to-speech
│   └── llm.py          # Claude API sentence formation
├── model/
│   ├── train.py        # Train the sign classifier
│   ├── signs.py        # Sign vocabulary definition
│   └── model.pkl       # Trained model (generated, not committed)
├── data_collection/
│   └── collect.py      # Record training data for each sign
└── utils/
    └── landmark_utils.py
```

---

## Features

- **Sign language translation** — MediaPipe Hands + trained classifier + TTS speaker
- **Wake word detection** — "Hey JARVIS" via Porcupine (offline) or Vosk
- **Voice commands** — Whisper STT for natural language commands
- **Computer vision** — YOLOv8n object and person detection
- **Personality engine** — Mood state machine (happy, curious, tired, alert)
- **NLP reasoning** — Claude API for contextual responses and sentence formation
- **Offline fallback** — ollama + Phi-3 mini for demos without internet

---

## Tech Stack

| Component | Technology |
|---|---|
| Platform | Raspberry Pi 4 + SunFounder PiDog SDK |
| Sign detection | MediaPipe Hands |
| Sign classification | scikit-learn RandomForest |
| Wake word | Picovoice Porcupine / Vosk |
| Speech-to-text | OpenAI Whisper |
| NLP / reasoning | Claude API (Anthropic) |
| Object detection | YOLOv8n + OpenCV |
| Text-to-speech | pyttsx3 / espeak-ng |
| Language | Python 3.11+ |

---

## Known Issues & Sprint Notes

### Issue #1 — SSH Disabled by Default on Raspberry Pi OS

**Sprint:** Sprint 2 — Hardware Assembly  
**Severity:** Blocker

**What happened:**  
When the SD card was flashed with a fresh Raspberry Pi OS image and the Pi was powered on for the first time, SSH access was unavailable. Current Raspberry Pi OS releases disable SSH by default as a security measure. The team did not have a spare monitor or keyboard available during the assembly session, so the Pi was completely inaccessible — no remote connection, no terminal, no ability to install software or run calibration scripts.

**Resolution:**  
Re-flashed the SD card using Raspberry Pi Imager with SSH explicitly enabled via the Advanced Options menu (`Ctrl+Shift+X`). WiFi credentials and hostname were also pre-configured at this stage, allowing the Pi to connect to the network and accept SSH connections on first boot.

**Prevention:**  
Pre-configuring the SD card through Raspberry Pi Imager's Advanced Options is now the standard first step in our setup process. A checklist entry has been added to the team's assembly guide to ensure this is never skipped.

---

### Issue #2 — Servo Misalignment Due to Skipped Zero Calibration

**Sprint:** Sprint 2 — Hardware Assembly  
**Severity:** High

**What happened:**  
During physical assembly of the PiDog kit, servo motors were attached to the leg brackets and linkage arms without first running the software calibration routine to set all servos to their 0° neutral position. The PiDog assembly geometry assumes every joint starts at a known zero angle — the servo horn and linkage arm positions are designed around this. Because the calibration step was skipped, several joints were assembled at incorrect offsets. On first boot, the dog was unable to stand level: multiple legs were visibly misaligned and the chassis sat at an angle rather than flat on all four feet.

**Resolution:**  
Disassembled the affected leg joints. Ran the servo zero routine via the SunFounder SDK to drive all servos to their neutral positions. Reattached all servo horns and linkage arms with joints correctly at 0°. After reassembly, the dog stood level and basic locomotion commands functioned correctly.

**Prevention:**  
Servo zeroing is now documented as a mandatory prerequisite in this README (see [Step 3 of Hardware Setup](#3-zero-all-servos-before-physical-assembly)) and added to the team's assembly checklist. No physical servo horn attachment should occur before this step is confirmed complete.

---

## Demo

🎥 Demo video — coming Sprint 3

---

## Base Repository

This project is forked from the official SunFounder PiDog repository:  
https://github.com/sunfounder/pidog

Full hardware documentation and wiring diagrams:  
https://docs.sunfounder.com/projects/pidog/en/latest/
