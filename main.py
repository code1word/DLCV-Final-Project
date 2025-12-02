import os
import cv2

from pose_extraction import HolisticPoseExtractor
from stick_figure_renderer import render_stick_figure
from emotion_classifier import EmotionClassifier
from face_utils import crop_face


# ---------------------------------------------------------------------
# MAIN - BATCH PROCESSING SCRIPT
# ---------------------------------------------------------------------
# Small driver script used for batch processing images from `data/`.
# It extracts MediaPipe keypoints, optionally crops faces for emotion
# classification, renders a stick figure overlay, and writes outputs
# under `pose_debug/` and `stick_output/`.


def main():
    # -----------------------------------------------------------------
    # DIRECTORIES
    # -----------------------------------------------------------------
    input_dir = "data"
    output_pose_dir = "pose_debug"
    output_stick_dir = "stick_output"

    os.makedirs(output_pose_dir, exist_ok=True)
    os.makedirs(output_stick_dir, exist_ok=True)

    # -----------------------------------------------------------------
    # INITIALIZE MODULES
    # -----------------------------------------------------------------
    extractor = HolisticPoseExtractor(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_face=True,
        enable_hands=True,
    )

    emotion_model = EmotionClassifier(model_path="emotion_model.pth")

    # -----------------------------------------------------------------
    # PROCESS EACH IMAGE
    # -----------------------------------------------------------------
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(input_dir, fname)
        image_bgr = cv2.imread(path)

        if image_bgr is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        print(f"Processing: {fname}")

        # -----------------------------------------------------------------
        # POSE + FACE EXTRACTION
        # -----------------------------------------------------------------
        results = extractor.extract(image_bgr)
        pose_pixels = results["pose"]["pixel"]

        # -----------------------------------------------------------------
        # FACE CROP (using MediaPipe mesh)
        # -----------------------------------------------------------------
        emotion = "neutral"  # fallback default

        # results["face"]["pixel"] is Nx3 (x, y, visibility)
        face_pixel_landmarks = results["face"]["pixel"]

        if face_pixel_landmarks.shape[0] > 0:
            # Use MediaPipe to get full landmark object for cropping
            # (MediaPipe holistic.process must be re-run to access raw landmarks)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_results = extractor.holistic.process(image_rgb)

            if mp_results.face_landmarks is not None:
                face_crop = crop_face(image_bgr, mp_results.face_landmarks)

                # Ensure crop is valid
                if face_crop is not None and face_crop.size > 0:
                    try:
                        emotion = emotion_model.predict(face_crop)
                        print(f"  Detected emotion: {emotion}")
                    except Exception as e:
                        print(f"[WARN] Emotion classifier failed: {e}")
                        emotion = "neutral"

        # -----------------------------------------------------------------
        # RENDER STICK FIGURE
        # -----------------------------------------------------------------
        stick = render_stick_figure(
            pose_pixels=pose_pixels,
            canvas_shape=(image_bgr.shape[0], image_bgr.shape[1]),
            emotion=emotion,
            thickness=15,
            background_bgr=image_bgr
        )

        cv2.imwrite(os.path.join(output_stick_dir, fname), stick)

        # -----------------------------------------------------------------
        # SAVE POSE DEBUG VISUALIZATION
        # -----------------------------------------------------------------
        debug_img = extractor.visualize_pose(image_bgr, show=False)
        cv2.imwrite(os.path.join(output_pose_dir, fname), debug_img)

    # -----------------------------------------------------------------
    # CLEANUP
    # -----------------------------------------------------------------
    extractor.close()
    print("Done!")


if __name__ == "__main__":
    main()
