import os
from typing import List, Tuple
import numpy as np
import cv2

class HaarFaceDetector:
    def __init__(self):
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(haar_path):
            raise RuntimeError("Không tìm thấy haarcascade_frontalface_default.xml")
        self.det = cv2.CascadeClassifier(haar_path)

    def align_face_by_eyes(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Placeholder để tương thích với RecognizeWorker.
        Ở đây chưa ước lượng mắt/align thực sự; chỉ cắt ROI và clip biên.
        """
        if gray is None or gray.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        H, W = gray.shape[:2]
        xs, ys = max(0, x), max(0, y)
        xe, ye = min(W, x + w), min(H, y + h)
        if xs >= xe or ys >= ye:
            return np.zeros((0, 0), dtype=np.uint8)
        return gray[ys:ye, xs:xe]

    def preprocess_face_gray(self, gray_crop: np.ndarray) -> np.ndarray:
        """
        Đầu vào/ra: ảnh GRAY.
        """
        if gray_crop.size == 0:
            return gray_crop
        g = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
        return g
    
    def detect(self, gray: np.ndarray) -> List[Tuple[int,int,int,int]]:
        faces = self.det.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]