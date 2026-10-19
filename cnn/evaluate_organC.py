import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist import OrganCMNIST, INFO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64
NUM_CLASSES = 11
BATCH_SIZE = 64

MODEL_PATH = (
    "cnn/models/"
    "organcmnist_fast_cnn_classifier.pth"
)

RESULT_PATH = (
    "cnn/results/"
    "organcmnist_evaluation.txt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("OrganCMNIST - Full Test Evaluation")
print("=" * 70)

print("Device     :", device)
print("Image size :", IMAGE_SIZE)
print("Batch size :", BATCH_SIZE)
print("Model      :", MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found!")
    print("Expected:")
    print(MODEL_PATH)

    raise FileNotFoundError(MODEL_PATH)


# ============================================================
# TRANSFORM
# MUST MATCH TRAINING
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Grayscale(
        num_output_channels=1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading OrganCMNIST test dataset...")

test_dataset = OrganCMNIST(
    split="test",
    transform=transform,
    download=True
)

print(
    "Full test images:",
    len(test_dataset)
)


# ============================================================
# DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# EXACT SAME ARCHITECTURE AS TRAINING
# ============================================================

class OrganCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(
                (1, 1)
            ),

            nn.Flatten(),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating OrganCNN model...")

model = OrganCNN()

model = model.to(device)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

print("Loading trained model...")

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# CLASS NAMES
# ============================================================

info = INFO["organcmnist"]

CLASS_NAMES = {
    int(key): value
    for key, value in info["label"].items()
}

print("\nOrganCMNIST Classes:")
print("-" * 70)

for class_id in range(NUM_CLASSES):

    print(
        f"{class_id}: "
        f"{CLASS_NAMES[class_id]}"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting evaluation...")
print("=" * 70)

all_labels = []
all_predictions = []

total_images = len(test_dataset)
processed_images = 0


model.eval()

with torch.no_grad():

    for batch_number, (
        images,
        labels
    ) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(device)

        labels = (
            labels
            .squeeze()
            .long()
            .to(device)
        )

        outputs = model(
            images
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        processed_images += images.size(0)

        if (
            batch_number % 50 == 0
            or processed_images == total_images
        ):

            print(
                f"Processed "
                f"{processed_images:,} / "
                f"{total_images:,}"
            )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    ),
    target_names=[
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
    ],
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(
        range(NUM_CLASSES)
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("OrganCMNIST Evaluation Results")
print("=" * 70)

print(
    f"Test Images : {total_images:,}"
)

print(
    f"Accuracy    : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision   : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall      : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score    : "
    f"{f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("Classification Report")
print("=" * 70)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("Confusion Matrix")
print("=" * 70)

print(confusion)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    "cnn/results",
    exist_ok=True
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "OrganCMNIST Evaluation Results\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Test Images : "
        f"{total_images:,}\n"
    )

    file.write(
        f"Accuracy    : "
        f"{accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision   : "
        f"{precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall      : "
        f"{recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score    : "
        f"{f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        str(confusion)
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("Evaluation completed successfully.")
print("=" * 70)

print(
    "Results saved to:"
)

print(RESULT_PATH)

print("=" * 70)