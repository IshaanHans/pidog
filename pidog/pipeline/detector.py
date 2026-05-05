import cv2
import numpy as np
import sys, os
sys.path.append(os.path.expanduser('~/pidog'))
from utils.landmark_utils import normalise_landmarks

class HandDetector:
    def __init__(self, model_complexity=0):
        from ultralytics import YOLO
        from picamera2 import Picamera2
        import time

        self.model = YOLO('yolov8n-pose.pt')

        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        self.picam.configure(config)
        self.picam.start()
        time.sleep(0.5)
        print('[Detector] YOLOv8 + picamera2 ready')

    def capture_frame(self):
        frame = self.picam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def process(self, bgr_frame=None):
        if bgr_frame is None:
            bgr_frame = self.capture_frame()

        results = self.model(bgr_frame, imgsz=320, verbose=False)
        annotated = results[0].plot()
        vector = None

        if results[0].keypoints is not None:
            kps = results[0].keypoints.data
            if len(kps) > 0:
                kp = kps[0].cpu().numpy()
                h, w = bgr_frame.shape[:2]
                kp[:, 0] /= w
                kp[:, 1] /= h
                vector = normalise_landmarks(kp.flatten().astype(np.float32))

        return vector, annotated

    def close(self):
        self.picam.stop()
        self.picam.close()
