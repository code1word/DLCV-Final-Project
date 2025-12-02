import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image


# ---------------------------------------------------------------------
# EMOTION CLASSIFIER (MobileNetV2)
# ---------------------------------------------------------------------
# Lightweight wrapper around a pretrained MobileNetV2-based emotion
# classifier. The class loads model weights (emotion_model.pth) and
# exposes a simple predict method that accepts a BGR face crop and
# returns one of the emotion labels.


class EmotionClassifier:
    def __init__(self, model_path="emotion_model.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classes = ["angry", "disgust", "fear",
                        "happy", "neutral", "sad", "surprise"]

        # -----------------------------------------------------------------
        # TRANSFORM PIPELINE (matches training)
        # -----------------------------------------------------------------
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        # -----------------------------------------------------------------
        # MODEL: MobileNetV2 (pretrained backbone, small classifier)
        # -----------------------------------------------------------------
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, len(self.classes))

        # -----------------------------------------------------------------
        # LOAD WEIGHTS
        # -----------------------------------------------------------------
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, face_bgr):
        # -----------------------------------------------------------------
        # PREDICT: Convert BGR -> RGB -> PIL and run inference
        # -----------------------------------------------------------------
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # -----------------------------------------------------------------
        # TRANSFORM TO TENSOR
        # -----------------------------------------------------------------
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # -----------------------------------------------------------------
        # INFERENCE
        # -----------------------------------------------------------------
        with torch.no_grad():
            logits = self.model(img_tensor)
            pred = logits.argmax(dim=1).item()

        return self.classes[pred]
