import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Dict, Optional


# ---------------------------------------------------------------------
# MEDIAPIPE / MODULE SETUP
# ---------------------------------------------------------------------
# Configure references to MediaPipe submodules used throughout this
# module. These are convenience handles so the rest of the code can
# reference these utilities without repeating long paths.
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------
# DATA TYPES: 2D LANDMARKS
# ---------------------------------------------------------------------
@dataclass
class Landmark2D:
    """Single 2D landmark in normalized coordinates."""
    x: float  # normalized [0, 1] from left to right
    y: float  # normalized [0, 1] from top to bottom
    visibility: float  # MediaPipe visibility score in [0, 1]


@dataclass
class Landmark2DPixel:
    """Single 2D landmark in pixel coordinates."""
    x: int
    y: int
    visibility: float


# ---------------------------------------------------------------------
# LANDMARKS HELPERS
# ---------------------------------------------------------------------
def _landmarks_to_arrays(
    landmarks,
    image_width: int,
    image_height: int
) -> Dict[str, np.ndarray]:
    """
    Convert MediaPipe landmarks into:
      - normalized array of shape (N, 3)
      - pixel array of shape (N, 3)
    Returns:
      {
        "normalized": np.ndarray,
        "pixel": np.ndarray
      }
    """
    if landmarks is None:
        return {
            "normalized": np.zeros((0, 3), dtype=np.float32),
            "pixel": np.zeros((0, 3), dtype=np.float32),
        }

    norm = []
    pix = []
    for lm in landmarks.landmark:
        x_norm = lm.x
        y_norm = lm.y
        vis = lm.visibility if hasattr(lm, "visibility") else 1.0

        x_px = int(x_norm * image_width)
        y_px = int(y_norm * image_height)

        norm.append([x_norm, y_norm, vis])
        pix.append([x_px, y_px, vis])

    return {
        "normalized": np.array(norm, dtype=np.float32),
        "pixel": np.array(pix, dtype=np.float32),
    }


# ---------------------------------------------------------------------
# HOLISTIC POSE EXTRACTOR
# ---------------------------------------------------------------------
class HolisticPoseExtractor:
    """Wrapper around MediaPipe Holistic for full-body keypoint extraction.

    Usage:
        extractor = HolisticPoseExtractor()
        result = extractor.extract(image_bgr)
        extractor.close()
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_face: bool = True,
        enable_hands: bool = True,
    ):
        self.enable_face = enable_face
        self.enable_hands = enable_hands

        self.holistic = mp_holistic.Holistic(
            static_image_mode=True,  # configured for still images
            model_complexity=2,
            smooth_landmarks=True,
            refine_face_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, image_bgr: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Run holistic inference on a single BGR image.

        Args:
            image_bgr: np.ndarray (H, W, 3) in BGR format (OpenCV style)

        Returns:
            {
              "pose": { "normalized": (33,3), "pixel": (33,3) },
              "face": { "normalized": (N,3),   "pixel": (N,3)   },
              "left_hand": { ... },
              "right_hand": { ... }
            }
            Any missing part will have (0, 3) arrays.
        """
        h, w, _ = image_bgr.shape

        # MediaPipe expects RGB input, convert from OpenCV's BGR
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        outputs = {}

        # -----------------------------------------------------------------
        # EXTRACT: Pose / Face / Hands
        # -----------------------------------------------------------------
        # Pose landmarks (33 body joints)
        outputs["pose"] = _landmarks_to_arrays(results.pose_landmarks, w, h)

        # Face landmarks (optional)
        if self.enable_face:
            outputs["face"] = _landmarks_to_arrays(results.face_landmarks, w, h)
        else:
            outputs["face"] = {
                "normalized": np.zeros((0, 3), dtype=np.float32),
                "pixel": np.zeros((0, 3), dtype=np.float32),
            }

        # Hands (optional)
        if self.enable_hands:
            outputs["left_hand"] = _landmarks_to_arrays(results.left_hand_landmarks, w, h)
            outputs["right_hand"] = _landmarks_to_arrays(results.right_hand_landmarks, w, h)
        else:
            outputs["left_hand"] = {
                "normalized": np.zeros((0, 3), dtype=np.float32),
                "pixel": np.zeros((0, 3), dtype=np.float32),
            }
            outputs["right_hand"] = {
                "normalized": np.zeros((0, 3), dtype=np.float32),
                "pixel": np.zeros((0, 3), dtype=np.float32),
            }

        return outputs

    def visualize_pose(
        self,
        image_bgr: np.ndarray,
        results_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        show: bool = True,
        window_name: str = "Holistic Pose"
    ) -> np.ndarray:
        """Draw pose landmarks on an image for debugging.

        If no precomputed results are provided, the method will run
        inference internally and use MediaPipe drawing utilities to
        overlay pose/face/hand landmarks on the image.

        Returns the BGR image with the drawn skeleton / landmarks.
        """
        draw_image = image_bgr.copy()
        h, w, _ = draw_image.shape

        if results_dict is None:
            # If the user didn't pass precomputed results, recompute
            image_rgb = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True
        else:
            # Reconstruct a fake results object just for drawing.
            image_rgb = cv2.cvtColor(draw_image, cv2.COLOR_BGR2RGB)
            results = mp_holistic.HolisticResults(
                pose_landmarks=None,
                face_landmarks=None,
                left_hand_landmarks=None,
                right_hand_landmarks=None,
                segmentation_mask=None,
            )
            image_rgb.flags.writeable = False
            results = self.holistic.process(image_rgb)
            image_rgb.flags.writeable = True

        # -----------------------------------------------------------------
        # DRAW: pose, face, hands
        # -----------------------------------------------------------------
        mp_drawing.draw_landmarks(
            draw_image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        if self.enable_face:
            mp_drawing.draw_landmarks(
                draw_image,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

        if self.enable_hands:
            mp_drawing.draw_landmarks(
                draw_image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )
            mp_drawing.draw_landmarks(
                draw_image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style()
            )

        if show:
            cv2.imshow(window_name, draw_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return draw_image

    def close(self):
        self.holistic.close()
