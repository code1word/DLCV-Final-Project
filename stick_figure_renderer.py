import cv2
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------
# POSE LANDMARK INDICES
# ---------------------------------------------------------------------
# MediaPipe-style pose landmark indices used by the renderer. These are
# small integer indices into the pose_pixels array (x, y, visibility).
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP = 23
R_HIP = 24

# ---------------------------------------------------------------------
# SKELETON CONNECTIONS
# ---------------------------------------------------------------------
# Lists of landmark index pairs describing standard limb connections.
ARM_CONNECTIONS = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

LEG_CONNECTIONS = [
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


# ---------------------------------------------------------------------
# BASIC DRAWING HELPERS
# ---------------------------------------------------------------------
def _draw_line(img, p1, p2, color, thickness):
    cv2.line(
        img,
        (int(p1[0]), int(p1[1])),
        (int(p2[0]), int(p2[1])),
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


def _draw_joint(img, center, radius, color):
    cv2.circle(
        img,
        (int(center[0]), int(center[1])),
        radius,
        color,
        thickness=-1,
        lineType=cv2.LINE_AA
    )


def _distance(p1, p2):
    """
    Euclidean distance between two 2D points.

    Input points are tuples/lists (x, y) or arrays. This is a tiny
    helper used by the renderer to compute distances for sizing.
    """

    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5


# ---------------------------------------------------------------------
# BODY MASK BUILDER (BACKGROUND CLEANUP)
# ---------------------------------------------------------------------
def _build_body_mask(pose_pixels: np.ndarray, H: int, W: int) -> np.ndarray:
    """Build a rough body mask from pose keypoints then dilate.

    This function constructs a convex hull across visible pose
    landmarks to get a rough silhouette, fills it, and dilates so the
    mask covers the person's body for stronger background cleanup
    operations like inpainting.
    """
    mask = np.zeros((H, W), dtype=np.uint8)

    if pose_pixels is None or pose_pixels.shape[0] == 0:
        return mask

    points = []
    for x, y, v in pose_pixels:
        if v > 0.3:  # visible point
            xi, yi = int(x), int(y)
            if 0 <= xi < W and 0 <= yi < H:
                points.append([xi, yi])

    if len(points) < 3:
        # Not enough points for a convex hull
        return mask

    pts = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(pts)

    cv2.fillConvexPoly(mask, hull, 255)

    # Dilate to cover a bit more than just joints (full body blob)
    k = max(25, int(min(H, W) * 0.06))  # scale kernel with image size
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ---------------------------------------------------------------------
# EMOTION FACE DISPATCHER
# ---------------------------------------------------------------------
def _draw_emotion_face(canvas, head_xy, size, emotion, outline_thickness):
    emotion = emotion.lower()

    if emotion == "happy":
        _draw_happy(canvas, head_xy, size, outline_thickness)
    elif emotion == "sad":
        _draw_sad(canvas, head_xy, size, outline_thickness)
    elif emotion == "angry":
        _draw_angry(canvas, head_xy, size, outline_thickness)
    elif emotion == "surprise":
        _draw_surprised(canvas, head_xy, size, outline_thickness)
    elif emotion == "fear":
        _draw_fear(canvas, head_xy, size, outline_thickness)
    elif emotion == "disgust":
        _draw_disgust(canvas, head_xy, size, outline_thickness)
    else:
        _draw_neutral(canvas, head_xy, size, outline_thickness)


# ---------------------------------------------------------------------
# FACE DRAWING FUNCTIONS
# ---------------------------------------------------------------------

# HAPPY
def _draw_happy(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    cv2.circle(canvas, (x - r // 3, y - r // 4), r // 8, (0, 0, 0), -1)
    cv2.circle(canvas, (x + r // 3, y - r // 4), r // 8, (0, 0, 0), -1)

    cv2.ellipse(canvas, (x, y + r // 6), (r // 3, r // 4), 0, 0, 180, (0, 0, 0), 8)


# SAD
def _draw_sad(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    cv2.circle(canvas, (x - r // 3, y - r // 4), r // 10, (0, 0, 0), -1)
    cv2.circle(canvas, (x + r // 3, y - r // 4), r // 10, (0, 0, 0), -1)

    cv2.ellipse(canvas, (x, y + r // 3), (r // 3, r // 6), 0, 180, 360, (0, 0, 0), 8)


# ANGRY
def _draw_angry(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    # Face circle
    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    # Eyebrows — longer & higher above eyes
    eyebrow_y_top = int(y - (r * 0.45))
    eyebrow_y_bottom = int(y - (r * 0.32))

    # Left eyebrow
    cv2.line(
        canvas,
        (int(x - r * 0.45), eyebrow_y_top),
        (int(x - r * 0.23), eyebrow_y_bottom),
        (0, 0, 0), 6
    )

    # Right eyebrow
    cv2.line(
        canvas,
        (int(x + r * 0.45), eyebrow_y_top),
        (int(x + r * 0.23), eyebrow_y_bottom),
        (0, 0, 0), 6
    )

    # Eyes
    eye_y = int(y - r * 0.25)
    cv2.circle(canvas, (int(x - r * 0.30), eye_y), int(r * 0.10), (0, 0, 0), -1)
    cv2.circle(canvas, (int(x + r * 0.30), eye_y), int(r * 0.10), (0, 0, 0), -1)

    # Angry mouth
    mouth_y = int(y + r * 0.30)
    cv2.line(
        canvas,
        (int(x - r * 0.33), mouth_y),
        (int(x + r * 0.33), mouth_y),
        (0, 0, 0), 8
    )


# SURPRISED
def _draw_surprised(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    # Dot eyes
    cv2.circle(canvas, (x - r // 3, y - r // 4), r // 10, (0, 0, 0), -1)
    cv2.circle(canvas, (x + r // 3, y - r // 4), r // 10, (0, 0, 0), -1)

    # Big O mouth
    cv2.circle(canvas, (x, y + r // 4), r // 5, (0, 0, 0), 4)


# NEUTRAL
def _draw_neutral(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    cv2.circle(canvas, (x - r // 3, y - r // 4), r // 10, (0, 0, 0), -1)
    cv2.circle(canvas, (x + r // 3, y - r // 4), r // 10, (0, 0, 0), -1)

    cv2.line(
        canvas,
        (x - r // 3, y + r // 3),
        (x + r // 3, y + r // 3),
        (0, 0, 0), 8
    )


# FEAR
def _draw_fear(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    # Base face
    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    # Eyes (wide, shocked)
    cv2.circle(
        canvas,
        (x - int(r * 0.33), y - int(r * 0.25)),
        int(r * 0.12),
        (0, 0, 0), 3
    )
    cv2.circle(
        canvas,
        (x + int(r * 0.33), y - int(r * 0.25)),
        int(r * 0.12),
        (0, 0, 0), 3
    )

    # Bigger vertical oval mouth
    cv2.ellipse(
        canvas,
        (x, y + int(r * 0.25)),
        (int(r * 0.18), int(r * 0.32)),
        0, 0, 360,
        (0, 0, 0), 4
    )


# DISGUST
def _draw_disgust(canvas, head_xy, size, thickness):
    x, y = int(head_xy[0]), int(head_xy[1])
    r = size

    cv2.circle(canvas, (x, y), r, (255, 255, 255), -1)
    cv2.circle(canvas, (x, y), r, (0, 0, 0), thickness)

    # Symmetrical angled eyebrows
    cv2.line(canvas, (x - r // 2, y - r // 3),
                     (x - r // 4, y - r // 4), (0, 0, 0), 6)
    cv2.line(canvas, (x + r // 2, y - r // 3),
                     (x + r // 4, y - r // 4), (0, 0, 0), 6)

    # Symmetrical half-closed eyelids
    cv2.ellipse(canvas, (x - r // 3, y - r // 5), (r // 10, r // 20), 0, 0, 180, (0, 0, 0), 6)
    cv2.ellipse(canvas, (x + r // 3, y - r // 5), (r // 10, r // 20), 0, 0, 180, (0, 0, 0), 6)

    # Wavy disgust mouth
    cv2.line(canvas, (x - r // 3, y + r // 3), (x - r // 6, y + r // 4), (0, 0, 0), 8)
    cv2.line(canvas, (x - r // 6, y + r // 4), (x + r // 6, y + r // 2), (0, 0, 0), 8)
    cv2.line(canvas, (x + r // 6, y + r // 2), (x + r // 3, y + r // 3), (0, 0, 0), 8)


# ---------------------------------------------------------------------
# STICK FIGURE RENDERING
# ---------------------------------------------------------------------
def render_stick_figure(
    pose_pixels: np.ndarray,
    canvas_shape: Tuple[int, int],
    emotion: str = "neutral",
    skeleton_color=(0, 0, 0),
    thickness=10,
    background_bgr=None,
) -> np.ndarray:

    H, W = canvas_shape

    # -------------------------------------------------------------
    # BACKGROUND PREPROCESSING:
    # 1) Remove body via inpainting
    # 2) Apply extreme blur
    # -------------------------------------------------------------
    if background_bgr is not None:
        bg = cv2.resize(background_bgr, (W, H))

        # Build body mask from pose keypoints
        body_mask = _build_body_mask(pose_pixels, H, W)

        if np.any(body_mask > 0):
            # Remove the figure using inpainting
            bg_no_body = cv2.inpaint(bg, body_mask, 25, cv2.INPAINT_TELEA)
        else:
            bg_no_body = bg

        # -------------------------------------------------------------
        # EXTREMELY BLURRED + BRIGHTENED BACKGROUND
        # -------------------------------------------------------------
        blurred = cv2.GaussianBlur(bg_no_body, (0, 0), sigmaX=80, sigmaY=80)

        # Increase brightness and contrast
        alpha = 1.35   # contrast (1.0 = no change, >1 increases contrast)
        beta  = 50     # brightness (0 = no change, positive lightens)

        bright = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)

        canvas = bright.copy()

    else:
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    joint_radius = max(5, thickness // 3)

    # -----------------------------------------------------------------
    # SHOULDERS MIDPOINT
    # -----------------------------------------------------------------
    if pose_pixels[L_SHOULDER][2] > 0.3 and pose_pixels[R_SHOULDER][2] > 0.3:
        shoulder = (
            (pose_pixels[L_SHOULDER][0] + pose_pixels[R_SHOULDER][0]) / 2,
            (pose_pixels[L_SHOULDER][1] + pose_pixels[R_SHOULDER][1]) / 2,
        )
    else:
        shoulder = pose_pixels[L_SHOULDER][:2]

    # -----------------------------------------------------------------
    # PELVIS MIDPOINT
    # -----------------------------------------------------------------
    if pose_pixels[L_HIP][2] > 0.3 and pose_pixels[R_HIP][2] > 0.3:
        hip_mid = (
            (pose_pixels[L_HIP][0] + pose_pixels[R_HIP][0]) / 2,
            (pose_pixels[L_HIP][1] + pose_pixels[R_HIP][1]) / 2,
        )
    else:
        hip_mid = pose_pixels[L_HIP][:2]

    pelvis = (hip_mid[0], hip_mid[1] - 25)

    # -----------------------------------------------------------------
    # ADAPTIVE HEAD SIZE
    # -----------------------------------------------------------------
    l_sh = pose_pixels[L_SHOULDER]
    r_sh = pose_pixels[R_SHOULDER]

    if l_sh[2] > 0.3 and r_sh[2] > 0.3:
        shoulder_width = _distance(l_sh, r_sh)
        head_size = int(0.9 * shoulder_width)
    else:
        torso_length = _distance(shoulder, pelvis)
        head_size = int(torso_length * 0.3)

    head_size = max(40, min(head_size, 250))
    neck_length = int(head_size * 0.25)

    # -----------------------------------------------------------------
     # HEAD POSITION BASED ON REAL FACE KEYPOINTS
     # -----------------------------------------------------------------

    # MediaPipe face ground-truth approximations
    # mouth_left = landmark 9, mouth_right = landmark 10
    chin_candidates = []
    for idx in [9, 10]:
        if pose_pixels[idx][2] > 0.3:  # visible enough
            chin_candidates.append(pose_pixels[idx][1])

    if len(chin_candidates) > 0:
        chin_y = np.mean(chin_candidates)
    else:
        # fallback if mouth isn't detected
        chin_y = shoulder[1] + head_size * 0.3

    # The bottom of the face circle should sit exactly at chin_y
    hx = shoulder[0]
    hy = chin_y - head_size  # center of head circle

    head = (hx, hy)
    neck = (hx, chin_y)


    # -----------------------------------------------------------------
    # SPINE
    # -----------------------------------------------------------------
    _draw_line(canvas, head, neck, skeleton_color, thickness)
    _draw_line(canvas, neck, shoulder, skeleton_color, thickness)
    _draw_line(canvas, shoulder, pelvis, skeleton_color, thickness)

    # -----------------------------------------------------------------
    # ARMS
    # -----------------------------------------------------------------
    for joint in [L_SHOULDER, R_SHOULDER]:
        x, y, v = pose_pixels[joint]
        if v > 0.3:
            _draw_line(canvas, shoulder, (x, y), skeleton_color, thickness)

    for i, j in ARM_CONNECTIONS:
        x1, y1, v1 = pose_pixels[i]
        x2, y2, v2 = pose_pixels[j]
        if v1 > 0.3 and v2 > 0.3:
            _draw_line(canvas, (x1, y1), (x2, y2), skeleton_color, thickness)

    # -----------------------------------------------------------------
    # LEGS
    # -----------------------------------------------------------------
    for hip in [L_HIP, R_HIP]:
        x, y, v = pose_pixels[hip]
        if v > 0.3:
            _draw_line(canvas, pelvis, (x, y), skeleton_color, thickness)

    for i, j in LEG_CONNECTIONS:
        x1, y1, v1 = pose_pixels[i]
        x2, y2, v2 = pose_pixels[j]
        if v1 > 0.3 and v2 > 0.3:
            _draw_line(canvas, (x1, y1), (x2, y2), skeleton_color, thickness)

    # -----------------------------------------------------------------
    # JOINTS
    # -----------------------------------------------------------------
    for idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
        x, y, v = pose_pixels[idx]
        if v > 0.3:
            _draw_joint(canvas, (x, y), joint_radius, skeleton_color)

    # -----------------------------------------------------------------
    # EMOTION FACE
    # -----------------------------------------------------------------
    _draw_emotion_face(
        canvas,
        head,
        size=head_size,
        emotion=emotion,
        outline_thickness=thickness,
    )

    return canvas
