from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarkerResult,
    RunningMode,
)
from mediapipe.tasks.python import BaseOptions

NUM_LANDMARKS = 21
FEATURES_PER_HAND = 63
TOTAL_FEATURES = 130

# Ruta por defecto al modelo de hand landmarker
_DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent / "data" / "hand_landmarker.task")

# Conexiones de la mano para dibujo (replica de mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (0, 9), (9, 10), (10, 11), (11, 12),   # Medio (middle)
    (0, 13), (13, 14), (14, 15), (15, 16), # Anular
    (0, 17), (17, 18), (18, 19), (19, 20), # Meñique
    (5, 9), (9, 13), (13, 17),             # Palma
]


class HandsFeatureExtractor:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | None = None,
    ) -> None:
        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No se encontró el modelo de hand landmarker: {model_path}\n"
                "Descárgalo de: https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    @staticmethod
    def _normalize_hand_landmarks(landmarks: List) -> np.ndarray:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        wrist = pts[0].copy()
        pts = pts - wrist
        scale = np.linalg.norm(pts[9])
        if scale < 1e-6:
            scale = 1.0
        pts /= scale
        return pts.reshape(-1)

    @staticmethod
    def _draw_hand_landmarks(
        frame: np.ndarray,
        landmarks: List,
        h: int,
        w: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        circle_radius: int = 3,
    ) -> None:
        """Dibuja los landmarks y las conexiones de una mano sobre el frame."""
        points = []
        for lm in landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), circle_radius, color, -1)

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], color, thickness)

    def extract_feature(self, bgr_frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result: HandLandmarkerResult = self.landmarker.detect(mp_image)
        feat = np.zeros((TOTAL_FEATURES,), dtype=np.float32)

        hand_data: Dict[str, np.ndarray] = {}
        handedness_score: Dict[str, float] = {"Left": -1.0, "Right": -1.0}

        h, w = bgr_frame.shape[:2]

        if result.hand_landmarks and result.handedness:
            for lm_list, hand_info in zip(result.hand_landmarks, result.handedness):
                label = hand_info[0].category_name   # "Left" / "Right"
                score = float(hand_info[0].score)
                hand_data[label] = self._normalize_hand_landmarks(lm_list)
                handedness_score[label] = score if label == "Right" else -score
                if draw:
                    color = (0, 255, 0) if label == "Right" else (255, 0, 0)
                    self._draw_hand_landmarks(bgr_frame, lm_list, h, w, color=color)

        if "Left" in hand_data:
            feat[0:63] = hand_data["Left"]
            feat[126] = 1.0
        if "Right" in hand_data:
            feat[63:126] = hand_data["Right"]
            feat[127] = 1.0

        feat[128] = handedness_score["Left"]
        feat[129] = handedness_score["Right"]
        return feat, bgr_frame

    def close(self) -> None:
        self.landmarker.close()
