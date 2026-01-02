# EmotiStick: Emotion-Aware Stick-Figure Abstraction from Photos

## Author: Dhruv Yalamanchi (Columbia University)

A compact end-to-end pipeline that turns a natural image of a person into a privacy-preserving stick-figure abstraction with an emotion-aware cartoon face. The system extracts pose and facial landmarks with MediaPipe Holistic, classifies facial emotion using a MobileNetV2 classifier (trained on FER2013), and renders a stylized stick figure with expressive, emoji-like facial drawings.

This project was built as a lightweight but complete demonstration of how classical computer vision, deep learning, and simple procedural graphics can be combined into a coherent visual abstraction pipeline.

---

## Features

- Pose & facial landmark extraction via MediaPipe Holistic
- Face cropping with MediaPipe mesh $\rightarrow$ emotion classification-ready crops
- MobileNetV2-based emotion classifier (happy, sad, angry, surprise, fear, disgust, neutral)
- Stick-figure renderer with proportional limb geometry, neck alignment, and emotion-driven facial icons
- Background abstraction (convex hull mask $\rightarrow$ inpainting $\rightarrow$ heavy blur $\rightarrow$ brightness boost)

- Utilities for:
  - Training a new emotion model (on FER2013)
  - Exporting emotion face icons
  - Batch-processing input photos

---

## Repository Structure

```
main.py                     # Batch driver for the full pipeline
pose_extraction.py          # MediaPipe Holistic wrapper
face_utils.py               # Face crop utilities
emotion_classifier.py       # Inference wrapper for MobileNetV2 model
train_emotion_model.py      # FER2013 training script
stick_figure_renderer.py    # Drawing code for stick figure + emotion faces
export_emotion_faces.py     # Generate standalone emotion face PNGs

data/                       # (user-provided) input images to process
face_examples/              # auto-generated example emotion faces
pose_debug/                 # MediaPipe landmark visualizations
stick_output/               # generated stick-figure outputs

emotion_model.pth           # (ignored in git) trained classifier weights

```

---

## Requirements

- Python 3.9+ (works with 3.8 in many environments)
- The project uses the following Python packages:
  - opencv-python
  - numpy
  - mediapipe
  - torch
  - torchvision
  - pillow

Install with pip (PowerShell / Windows example):

```powershell
python -m pip install --upgrade pip
pip install opencv-python numpy mediapipe torch torchvision pillow
```

Tip: Installing torch should follow the instructions for your platform/GPU at https://pytorch.org/. The simple pip line above will install CPU-only torch on many machines.

---

## Quickstart: Run the Demo (Batch Processing)

Place any photos you want to process into the `data/` folder (supported extensions: .jpg, .jpeg, .png).

Then run:

```powershell
python main.py
```

Outputs:

- `stick_output/` — resulting stick-figure images
- `pose_debug/` — MediaPipe visualizations of landmarks

If the repo already contains a trained model named `emotion_model.pth`, the script will use it. If not, follow the training steps below.

---

## Training (Optional)

The repository includes a training script `train_emotion_model.py` which trains a MobileNetV2 classifier on the FER2013 dataset. This is useful if you'd like to re-train or update the model.

Train using (this is CPU-friendly but slow — use a machine with GPU if you can):

```powershell
python train_emotion_model.py
```

Notes:

- The script expects a folder layout like:

```
fer2013/
  train/<class_name>/*.png
  test/<class_name>/*.png
```

- By default `train_emotion_model.py` subsamples to smaller train/test subsets to make CPU training feasible. See constants at the top of the script to change batch size, epochs and other config.

After training the model will be saved to `emotion_model.pth` which is used by the `EmotionClassifier` wrapper.

---

## Generating Face Drawing Examples

If you want just the face drawings exported to separate image files, run:

```powershell
python export_emotion_faces.py
```

The script writes `face_examples/<emotion>.png` for each supported emotion.

---

## Tips / Notes

- The code uses MediaPipe Holistic (33 pose landmarks + face mesh + hands). Behavior depends on the quality of the input images
- `render_stick_figure` supports an optional background image (the original image is blurred and brightened while the person is inpainted out for a simple aesthetic)
- Many scripts are intentionally simple and self-contained — they are a good starting point for research/visualization demos
