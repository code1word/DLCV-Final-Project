import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import random

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

DATA_DIR = "fer2013"      # Root directory containing train/ and test/ folders
BATCH_SIZE = 16
NUM_CLASSES = 7            # FER2013: angry, disgust, fear, happy, neutral, sad, surprise
EPOCHS = 10
LR = 1e-3                  # Adam learning rate
SUBSET_SIZE = 8000         # Train on 8k images for CPU feasibility
TEST_SUBSET_SIZE = 5000    # Evaluate on 5k images
DEVICE = "cpu" 


# ---------------------------------------------------------------------
# HELPER: Randomly subsample a torchvision dataset
# ---------------------------------------------------------------------
def random_subset(dataset, subset_size):
    """
    Return a random subset of a torchvision ImageFolder dataset.
    This modifies dataset.samples in-place to contain only subset_size items.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:subset_size]

    # Keep only selected image/label tuples
    dataset.samples = [dataset.samples[i] for i in indices]
    return dataset


# ---------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------
def main():

    # -------------------------------------------------------------
    # TRANSFORM PIPELINE:
    # Convert FER2013 to the same format expected by MobileNetV2:
    #   - Grayscale -> 3 channels (MobileNet expects 3)
    #   - Resize to 128 x 128
    #   - Normalize to [-1, 1]
    # -------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # -------------------------------------------------------------
    # LOAD DATASET
    # ImageFolder expects:
    #   fer2013/train/<class>/*.png
    #   fer2013/test/<class>/*.png
    # -------------------------------------------------------------
    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=transform)
    test_ds  = datasets.ImageFolder(f"{DATA_DIR}/test",  transform=transform)

    # -------------------------------------------------------------
    # SUBSAMPLE DATASET FOR FAST CPU TRAINING
    # -------------------------------------------------------------
    train_ds = random_subset(train_ds, SUBSET_SIZE)
    test_ds  = random_subset(test_ds, TEST_SUBSET_SIZE)

    print(f"Training on subset: {len(train_ds)} images")
    print(f"Testing on subset:  {len(test_ds)} images\n")

    # -------------------------------------------------------------
    # DATA LOADERS
    # num_workers=4 works on CPU for speed
    # pin_memory improves throughput on some machines
    # -------------------------------------------------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )

    # -------------------------------------------------------------
    # LOAD PRETRAINED MOBILENETV2
    # Replace the final classifier layer
    #
    # MobileNetV2 Architecture:
    #   features -> classifier[0] -> classifier[1]
    #
    # We replace classifier[1] with a new Linear layer with 7 outputs.
    # -------------------------------------------------------------
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model.to(DEVICE)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------------------------------------------------
    # TRAINING LOOP
    # -----------------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)       # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()             # Backprop
            optimizer.step()            # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {avg_loss:.4f}")

    # -----------------------------------------------------------------
    # SAVE TRAINED WEIGHTS
    # -----------------------------------------------------------------
    torch.save(model.state_dict(), "emotion_model.pth")
    print("\nSaved emotion_model.pth\n")

    # -----------------------------------------------------------------
    # QUICK EVALUATION ON TEST SUBSET
    # -----------------------------------------------------------------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.3f}")


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
