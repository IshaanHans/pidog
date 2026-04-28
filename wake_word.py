import os, threading, struct, time
from typing import Callable

class PorcupineDetector:
    def __init__(self, access_key=None, ppn_path=None,
                 keyword='jarvis', sensitivity=0.6, on_wake=None):
        import pvporcupine, pyaudio
        self.on_wake = on_wake
        self._running = False
        access_key = access_key or os.environ.get('PORCUPINE_ACCESS_KEY')
        if not access_key:
            raise ValueError("Set PORCUPINE_ACCESS_KEY env variable.")
        if ppn_path and os.path.exists(ppn_path):
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[ppn_path],
                sensitivities=[sensitivity])
        else:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity])
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=self.porcupine.sample_rate, channels=1,
            format=pyaudio.paInt16, input=True,
            frames_per_buffer=self.porcupine.frame_length)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        print('[WakeWord] Listening...')
        while self._running:
            pcm = self._stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from(f'{self.porcupine.frame_length}h', pcm)
            if self.porcupine.process(pcm) >= 0:
                print('[WakeWord] *** Detected! ***')
                if self.on_wake:
                    self.on_wake()

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()
        self.porcupine.delete()


class WakeWordDetector:
    def __init__(self, on_wake=None, ppn_path=None, sensitivity=0.6):
        self._detector = None
        key = os.environ.get('PORCUPINE_ACCESS_KEY')
        if key:
            try:
                self._detector = PorcupineDetector(
                    access_key=key, ppn_path=ppn_path,
                    sensitivity=sensitivity, on_wake=on_wake)
                self._backend = 'porcupine'
                return
            except Exception as e:
                print(f'[WakeWord] Porcupine failed: {e}')
        raise RuntimeError("No wake word backend available. Install pvporcupine and set PORCUPINE_ACCESS_KEY.")

    def start(self):
        if self._detector:
            self._detector.start()

    def stop(self):
        if self._detector:
            self._detector.stop()
