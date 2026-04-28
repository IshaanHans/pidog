import pickle, os, sys
from collections import deque
import numpy as np
sys.path.append(os.path.expanduser('~/pidog'))
from model.signs import INDEX_TO_SIGN

DEFAULT_MODEL = os.path.expanduser('~/pidog/model/model.pkl')

class SignClassifier:
    def __init__(self, model_path=DEFAULT_MODEL, confidence_threshold=0.75,
                 buffer_size=10, confirm_threshold=7, cooldown_frames=20):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run model/train.py first.")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.confirm_threshold = confirm_threshold
        self.cooldown_frames = cooldown_frames
        self._cooldown_counter = 0
        self._last_confirmed = None

    def predict(self, vector):
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
        probs = self.model.predict_proba([vector])[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob >= self.confidence_threshold:
            self.buffer.append(best_idx)
        else:
            self.buffer.append(None)
        if len(self.buffer) < self.buffer.maxlen:
            return None
        valid = [x for x in self.buffer if x is not None]
        if not valid:
            return None
        most_common = max(set(valid), key=valid.count)
        if valid.count(most_common) >= self.confirm_threshold:
            sign = INDEX_TO_SIGN.get(most_common)
            if sign == self._last_confirmed and self._cooldown_counter > 0:
                return None
            self._last_confirmed = sign
            self._cooldown_counter = self.cooldown_frames
            self.buffer.clear()
            return sign
        return None

    def get_top_prediction(self, vector):
        probs = self.model.predict_proba([vector])[0]
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        if best_prob >= self.confidence_threshold:
            return INDEX_TO_SIGN.get(best_idx), best_prob
        return None, best_prob
