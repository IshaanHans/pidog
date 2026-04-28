import argparse, os, sys, time, threading
import cv2
sys.path.append(os.path.expanduser('~/pidog'))
from pipeline.detector import HandDetector
from pipeline.classifier import SignClassifier
from pipeline.tts import Speaker

FONT = cv2.FONT_HERSHEY_SIMPLEX
ACTIVE_WINDOW_SECONDS = 10

class JarvisPipeline:
    def __init__(self, args):
        self.args = args
        self.state = 'IDLE'
        self._active_timer = None
        self._lock = threading.Lock()
        print('Initialising J.A.R.V.I.S...')
        self.speaker = Speaker(rate=155)
        self.detector = HandDetector(model_complexity=0)
        try:
            self.classifier = SignClassifier()
        except FileNotFoundError as e:
            print(f'ERROR: {e}')
            sys.exit(1)
        self.llm = None
        if args.llm:
            try:
                from pipeline.llm import LLMSentenceFormer
                self.llm = LLMSentenceFormer(buffer_size=5)
                print('  Claude API: ready')
            except Exception as e:
                print(f'  WARNING: LLM unavailable ({e})')
        self.wake_detector = None
        if not args.always_on:
            try:
                from pipeline.wake_word import WakeWordDetector
                self.wake_detector = WakeWordDetector(on_wake=self._on_wake)
            except Exception as e:
                print(f'WARNING: Wake word unavailable ({e}). Using always-on.')
                args.always_on = True
        self.confirmed_signs = []

    def _on_wake(self):
        with self._lock:
            if self.state == 'ACTIVE':
                self._reset_timer()
                return
            self.state = 'ACTIVE'
            self.confirmed_signs.clear()
        self.speaker.say('Yes?')
        self._reset_timer()

    def _reset_timer(self):
        if self._active_timer:
            self._active_timer.cancel()
        self._active_timer = threading.Timer(ACTIVE_WINDOW_SECONDS, self._go_idle)
        self._active_timer.daemon = True
        self._active_timer.start()

    def _go_idle(self):
        with self._lock:
            if self.state == 'IDLE':
                return
            self.state = 'IDLE'
            print('[STATE] -> IDLE')
            if self.llm and self.llm.buffer_contents:
                sentence = self.llm.flush()
                if sentence:
                    self.speaker.say(sentence)

    def run(self):
        if self.args.always_on:
            print('[MODE] Always-on')
            self.state = 'ACTIVE'
        else:
            print("[MODE] Say 'Hey JARVIS' to activate")
            self.wake_detector.start()
        try:
            while True:
                frame = self.detector.capture_frame()
                frame = cv2.flip(frame, 1)
                if self.state == 'ACTIVE':
                    vector, annotated = self.detector.process(frame)
                    if vector is not None:
                        confirmed = self.classifier.predict(vector)
                        if confirmed:
                            print(f'[SIGN] {confirmed}')
                            self.confirmed_signs.append(confirmed)
                            self._reset_timer()
                            if self.llm:
                                sentence = self.llm.add_sign(confirmed)
                                if sentence:
                                    self.speaker.say(sentence)
                            else:
                                self.speaker.say(confirmed.replace('_', ' ').lower())
                else:
                    annotated = frame
                if not self.args.no_display:
                    state_color = (0, 200, 80) if self.state == 'ACTIVE' else (80, 80, 80)
                    cv2.putText(annotated, f'JARVIS: {self.state}',
                                (10, 30), FONT, 0.8, state_color, 2)
                    signs_text = ' | '.join(self.confirmed_signs[-5:])
                    cv2.putText(annotated, signs_text,
                                (10, 60), FONT, 0.6, (244, 162, 97), 2)
                    cv2.putText(annotated, 'Q=quit  W=wake  C=clear',
                                (10, annotated.shape[0] - 10), FONT, 0.45, (60, 60, 60), 1)
                    cv2.imshow('J.A.R.V.I.S', annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('w'):
                        self._on_wake()
                    elif key == ord('c'):
                        self.confirmed_signs.clear()
                        if self.llm:
                            self.llm._buffer.clear()
        except KeyboardInterrupt:
            print('\nInterrupted.')
        finally:
            if self._active_timer:
                self._active_timer.cancel()
            if self.wake_detector:
                self.wake_detector.stop()
            cv2.destroyAllWindows()
            self.detector.close()
            self.speaker.wait_until_done()
            self.speaker.close()
            print('Shutdown complete.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', action='store_true')
    parser.add_argument('--always-on', action='store_true')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--ppn', type=str, default=None)
    args = parser.parse_args()
    JarvisPipeline(args).run()

if __name__ == '__main__':
    main()
