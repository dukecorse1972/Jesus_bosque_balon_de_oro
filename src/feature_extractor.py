from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np

NUM_LANDMARKS = 21
FEATURES_PER_HAND = 63
TOTAL_FEATURES = 130


@dataclass
class FrameFeatureResult:
    feature: np.ndarray
    fps: float


class HandsFeatureExtractor:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

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

    def extract_feature(self, bgr_frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        feat = np.zeros((TOTAL_FEATURES,), dtype=np.float32)

        hand_data: Dict[str, np.ndarray] = {}
        handedness_score: Dict[str, float] = {"Left": -1.0, "Right": -1.0}

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm_set, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = hand_info.classification[0].label  # Left / Right
                score = float(hand_info.classification[0].score)
                hand_data[label] = self._normalize_hand_landmarks(lm_set.landmark)
                handedness_score[label] = score if label == "Right" else -score
                if draw:
                    self.mp_draw.draw_landmarks(
                        bgr_frame,
                        lm_set,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

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
        self.hands.close()
