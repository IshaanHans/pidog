



import threading, queue
import pyttsx3

class Speaker:
    def __init__(self, rate=160, volume=1.0):
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._rate = rate
        self._volume = volume
        self._thread.start()

    def _worker(self):
        engine = pyttsx3.init()
        engine.setProperty('rate', self._rate)
        engine.setProperty('volume', self._volume)
        while True:
            text = self._queue.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            self._queue.task_done()

    def say(self, text):
        print(f'[TTS] → {text}')
        self._queue.put(text)

    def wait_until_done(self):
        self._queue.join()

    def close(self):
        self._queue.put(None)
        self._thread.join()
