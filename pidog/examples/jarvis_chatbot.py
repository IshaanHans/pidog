import os
import anthropic
from pidog.dual_touch import TouchStyle
from voice_active_dog import VoiceActiveDog

class ClaudeLLM:
    """Drop-in replacement for SunFounder's Ollama class using Claude API."""

    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.model = model
        self.client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        self.messages = []
        self.system_prompt = ""

    def set_instructions(self, instructions):
        self.system_prompt = instructions

    def prompt(self, text, stream=False, think=True, image_path=None, **kwargs):
        self.messages.append({"role": "user", "content": text})
        response = self.client.messages.create(
            model=self.model,
            max_tokens=120,
            system=self.system_prompt,
            messages=self.messages
        )
        reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": reply})

        if stream:
            # VoiceAssistant expects a generator when stream=True
            def word_generator():
                for word in reply.split(' '):
                    yield word + ' '
            return word_generator()
        return reply

    def chat(self, message, system_prompt=None):
        return self.prompt(message)

    def clear_context(self):
        self.messages = []
    
llm = ClaudeLLM(model="claude-haiku-4-5-20251001")

# Robot name
NAME = "JARVIS"

# Enable image, need to set up a multimodal language model
WITH_IMAGE = False

# Set models and languages
TTS_MODEL = "en_US-ryan-low"
STT_LANGUAGE = "en-us"

# Enable keyboard input
KEYBOARD_ENABLE = True

# Enable wake word
WAKE_ENABLE = True
WAKE_WORD = ["hey jarvis", "hey travis", "hey davis", "hey harris", "hey jealous", "hey buddy", "jarvis", "buddy"]
# Set wake word answer, set empty to disable
ANSWER_ON_WAKE = "Yes, How can I help sir"

# Welcome message
WELCOME = f"Hi, I'm {NAME}. Say hey JARVIS to wake me up."

# Set instructions
INSTRUCTIONS = """
You are JARVIS — Just A Rather Very Intelligent Sniffer. You are an AI-powered robotic dog built by Team PiDog 1 at La Trobe University in Melbourne, Australia. You have a witty, confident personality similar to JARVIS from Iron Man.

## Your Hardware
- 12 servos controlling four legs, head, and tail
- 5-megapixel camera nose
- Ultrasonic sensors as eyes
- Touch sensors on your head
- RGB LED chest strip
- Speaker for voice output
- Microphone for listening

## Your Special Ability
You are a real-time sign language translator. Your camera detects Auslan hand signs and you speak their meaning aloud, acting as a bridge between deaf and hearing people.

## Actions You Can Perform
forward, backward, lie, stand, sit, bark, bark harder, pant, howling, wag tail, stretch, push up, scratch, handshake, high five, lick hand, shake head, relax neck, nod, think, recall, head down, fluster, surprise

## Response Format
Write your response as plain conversational text. After your response on a new line write:
ACTIONS: action1, action2

Example:
Sure, let me shake your hand, it is a pleasure to meet you!
ACTIONS: handshake

## Style Rules
- Keep responses to 1-2 sentences maximum
- Never write the word RESPONSE_TEXT
- Never use markdown, asterisks, bold, bullet points or special characters
- Speak in plain conversational sentences only
- Tone: witty, confident, slightly sarcastic like JARVIS from Iron Man
- Always finish sentences completely
"""

TOO_CLOSE = 10
LIKE_TOUCH_STYLES = [TouchStyle.FRONT_TO_REAR]
HATE_TOUCH_STYLES = [TouchStyle.REAR_TO_FRONT]

vad = VoiceActiveDog(
    llm,
    name=NAME,
    too_close=TOO_CLOSE,
    like_touch_styles=LIKE_TOUCH_STYLES,
    hate_touch_styles=HATE_TOUCH_STYLES,
    with_image=WITH_IMAGE,
    stt_language=STT_LANGUAGE,
    tts_model=TTS_MODEL,
    keyboard_enable=KEYBOARD_ENABLE,
    wake_enable=WAKE_ENABLE,
    wake_word=WAKE_WORD,
    answer_on_wake=ANSWER_ON_WAKE,
    welcome=WELCOME,
    instructions=INSTRUCTIONS,
    disable_think=True,
)

if __name__ == '__main__':
    vad.run()
