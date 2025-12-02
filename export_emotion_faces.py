import cv2
import numpy as np
import os

# ---------------------------------------------------
# IMPORT YOUR FACE-DRAWING FUNCTIONS
# ---------------------------------------------------
from stick_figure_renderer import (
    _draw_happy,
    _draw_sad,
    _draw_angry,
    _draw_surprised,
    _draw_fear,
    _draw_disgust,
    _draw_neutral
)

FACE_FUNCS = {
    "happy": _draw_happy,
    "sad": _draw_sad,
    "angry": _draw_angry,
    "surprise": _draw_surprised,
    "fear": _draw_fear,
    "disgust": _draw_disgust,
    "neutral": _draw_neutral,
}

def crop_to_content(img, pad=10):
    """
    Crops the image to the min bounding box containing all non-white pixels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 250  # treat near-white as background

    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return img  # fallback if something goes wrong

    x, y, w, h = cv2.boundingRect(coords)

    # Add padding
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + w + pad, img.shape[1])
    y1 = min(y + h + pad, img.shape[0])

    return img[y0:y1, x0:x1]


def render_face_only(emotion, output_folder="face_examples"):
    """
    Render only the head + facial expression with tight cropping.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Temporary large canvas so nothing gets cut off
    H, W = 800, 800
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    head_xy = (W // 2, H // 2)
    radius = 200
    thickness = 12

    # Draw the face
    FACE_FUNCS[emotion](canvas, head_xy, radius, thickness)

    # Crop out unused whitespace
    cropped = crop_to_content(canvas)

    # Save
    outfile = os.path.join(output_folder, f"{emotion}.png")
    cv2.imwrite(outfile, cropped)
    print(f"[Saved] {outfile}")


def main():
    emotions = ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    for emo in emotions:
        render_face_only(emo)

if __name__ == "__main__":
    main()
