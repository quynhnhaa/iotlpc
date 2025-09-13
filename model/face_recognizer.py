import os
import glob
import numpy as np
import cv2
from typing import Dict, List, Tuple

# NOTE: Chuyển sang dùng OpenCV LBPHFaceRecognizer nhưng GIỮ NGUYÊN TÊN HÀM BÊN NGOÀI.
# Một số hàm sẽ trở thành wrapper để tương thích với luồng sẵn có.

class LBPFaceRecognizer:
    def __init__(self, db_dir: str = "faces_db"):
        # Tạo LBPH model (OpenCV)
        # Tham số tương ứng: radius=1, neighbors=8, grid_x=6, grid_y=6
        if not hasattr(cv2, 'face'):
            raise RuntimeError("OpenCV không có module 'cv2.face'. Cài đặt opencv-contrib-python.")
        self.model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=6, grid_y=6)
        self.db_dir = db_dir
        self.label_to_name: List[str] = []
        self.name_to_label: Dict[str, int] = {}
        self._trained = False
        self._last_image = None  # lưu ảnh xám 96x96 gần nhất để predict trong recognize_hist
        self._last_train_signature = None
        self._ensure_trained()

    # ---------- (legacy) Histogram distance API giữ nguyên để tương thích ----------
    def chi2_distance(self, a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
        # Không còn dùng cho nhận diện chính, nhưng giữ để không phá vỡ code cũ
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(0.5 * np.sum(((a - b) ** 2) / (a + b + eps)))

    def _dataset_signature(self) -> Tuple[int, int]:
        """Trả về chữ ký đơn giản của dữ liệu train (số người, tổng số ảnh) để quyết định retrain."""
        if not os.path.isdir(self.db_dir):
            return (0, 0)
        people = [d for d in os.listdir(self.db_dir) if os.path.isdir(os.path.join(self.db_dir, d))]
        total = 0
        for p in people:
            total += len(glob.glob(os.path.join(self.db_dir, p, '*.jpg'))) + \
                     len(glob.glob(os.path.join(self.db_dir, p, '*.png')))
        return (len(people), total)

    def _ensure_trained(self):
        sig = self._dataset_signature()
        if self._trained and self._last_train_signature == sig:
            return
        images = []
        labels = []
        self.label_to_name = []
        self.name_to_label = {}
        if not os.path.isdir(self.db_dir):
            os.makedirs(self.db_dir, exist_ok=True)
        people = sorted([d for d in os.listdir(self.db_dir) if os.path.isdir(os.path.join(self.db_dir, d))])
        for li, name in enumerate(people):
            self.name_to_label[name] = li
            self.label_to_name.append(name)
            for ext in ('*.jpg', '*.png', '*.jpeg', '*.bmp'):
                for p in glob.glob(os.path.join(self.db_dir, name, ext)):
                    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if img is None or img.size == 0:
                        continue
                    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)
                    images.append(img)
                    labels.append(li)
        if images:
            try:
                self.model.train(images, np.array(labels, dtype=np.int32))
                self._trained = True
            except Exception:
                # Nếu train lỗi, coi như chưa train
                self._trained = False
        else:
            self._trained = False
        self._last_train_signature = sig

    def recognize_hist(self, emb: np.ndarray, db_emb: Dict[str, List[List[float]]],
                       thresh: float = 0.60, margin: float = 0.03) -> Tuple[str, float, float]:
        """
        Thay vì so khớp histogram tự code, dùng LBPHFaceRecognizer.predict trên ảnh xám 96x96 gần nhất.
        Vẫn trả về (label, best, second) để tương thích.
        """
        # Đảm bảo model đã train (reload nếu có dữ liệu mới)
        self._ensure_trained()
        if not self._trained or self._last_image is None:
            return "unknown", 1e9, 1e9
        try:
            pred_label, confidence = self.model.predict(self._last_image)
        except Exception:
            return "unknown", 1e9, 1e9
        name = self.label_to_name[pred_label] if 0 <= pred_label < len(self.label_to_name) else "unknown"
        # Với LBPH, confidence càng nhỏ càng tốt. Chọn ngưỡng kinh nghiệm.
        # Map ngưỡng từ [0..100] ~ tốt; điều chỉnh nếu cần qua tham số thresh (dùng như scale).
        conf_thresh = max(30.0, 100.0 * thresh)  # nếu thresh=0.6 => 60.0
        if confidence <= conf_thresh and name != "unknown":
            # second-best không có sẵn từ API; ước lượng bằng best + margin
            return name, float(confidence), float(confidence + max(1e-3, margin))
        return "unknown", float(confidence), float(confidence + max(1e-3, margin))

    # ---------- Wrapper giữ nguyên tên để lấy ảnh đầu vào cho LBPH ----------
    def lbp_u8(self, image: np.ndarray) -> np.ndarray:
        """
        Giữ hàm để không phá vỡ import, nhưng giờ chỉ chuẩn hóa ảnh về 96x96 và trả về chính ảnh.
        """
        if image is None or image.size == 0:
            return image
        g = image.astype(np.uint8)
        g = cv2.resize(g, (96, 96), interpolation=cv2.INTER_LINEAR)
        return g

    def lbp_grid_hist(self, gray_crop: np.ndarray, grid=(6,6)) -> np.ndarray:
        """
        GIỮ NGUYÊN TÊN HÀM. Bây giờ: chuẩn hóa ảnh về 96x96, lưu lại vào self._last_image để recognize_hist dùng.
        Trả về một vector dummy (1,) để không làm hỏng các chỗ đang lưu 'emb' vào DB, nhưng sẽ KHÔNG dùng để nhận diện.
        """
        if gray_crop is None or gray_crop.size == 0:
            self._last_image = None
            return np.zeros((1,), dtype=np.float32)
        roi = cv2.resize(gray_crop, (96, 96), interpolation=cv2.INTER_LINEAR)
        self._last_image = roi
        return np.zeros((1,), dtype=np.float32)  # dummy embedding không sử dụng