# ---------------------------------------------------------------------
# FACE CROPPING UTILITIES
# ---------------------------------------------------------------------
def crop_face(image_bgr, face_landmarks):
    """Crop a face region from a BGR image using MediaPipe face landmarks.

    The function computes the min/max of normalized landmark (x,y) values,
    converts them to pixel coordinates, expands the rectangle slightly for
    padding, clamps to the image bounds, and returns the cropped BGR patch.

    Args:
        image_bgr: np.ndarray (H, W, 3) OpenCV BGR image
        face_landmarks: MediaPipe face landmarks object with landmark list

    Returns:
        np.ndarray: cropped BGR image region (may be empty if landmarks
                    are invalid/out-of-bounds)
    """

    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]

    h, w, _ = image_bgr.shape

    # Convert normalized coordinates to pixels
    x1 = int(min(xs) * w)
    y1 = int(min(ys) * h)
    x2 = int(max(xs) * w)
    y2 = int(max(ys) * h)

    # Add padding (px) and clamp to image bounds
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    return image_bgr[y1:y2, x1:x2]
