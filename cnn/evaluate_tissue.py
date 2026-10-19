import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import TissueMNIST, INFO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 64
NUM_CLASSES = 8
BATCH_SIZE = 64

MODEL_PATH = (
    "cnn/models/"
    "tissuemnist_fast_cnn_classifier.pth"
)

RESULT_PATH = (
    "cnn/results/"
    "tissuemnist_evaluation.txt"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("TissueMNIST - Model Evaluation")
print("=" * 60)

print("Device:", device)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)


# ============================================================
# TRANSFORM
# Must match training
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5],
        std=[0.5]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading TissueMNIST test dataset...")

test_dataset = TissueMNIST(
    split="test",
    transform=transform,
    download=True
)

print("Test images:", len(test_dataset))


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# Must be exactly the same architecture as training
# ============================================================

class TissueCNN(nn.Module):

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

            nn.AdaptiveAvgPool2d((1, 1)),

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

print("\nCreating model...")

model = TissueCNN()

model = model.to(device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# GET CLASS NAMES
# ============================================================

info = INFO["tissuemnist"]

CLASS_NAMES = {
    int(k): v
    for k, v in info["label"].items()
}

print("\nClasses:")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"{class_id}: {class_name}"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\nStarting evaluation...")
print("=" * 60)

all_predictions = []
all_labels = []

correct = 0
total = 0


with torch.no_grad():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        labels = labels.squeeze().long().to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        if (batch_index + 1) % 100 == 0:

            print(
                f"Processed "
                f"{min((batch_index + 1) * BATCH_SIZE, total)} "
                f"/ {len(test_dataset)} images"
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
    labels=list(range(NUM_CLASSES)),
    target_names=[
        CLASS_NAMES[i]
        for i in range(NUM_CLASSES)
    ],
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TissueMNIST Evaluation Results")
print("=" * 60)

print(
    f"Test Images : {len(test_dataset)}"
)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1 Score    : {f1 * 100:.2f}%"
)


print("\nClassification Report:")
print(report)


print("Confusion Matrix:")
print(cm)


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
        "TissueMNIST Evaluation Results\n"
    )

    file.write(
        "=" * 60 + "\n\n"
    )

    file.write(
        f"Test Images : {len(test_dataset)}\n"
    )

    file.write(
        f"Accuracy    : {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Precision   : {precision * 100:.2f}%\n"
    )

    file.write(
        f"Recall      : {recall * 100:.2f}%\n"
    )

    file.write(
        f"F1 Score    : {f1 * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report:\n"
    )

    file.write(
        report
    )

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(cm)
    )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 60)
print("Evaluation completed.")
print("=" * 60)

print(
    "Results saved to:"
)

print(
    RESULT_PATH
)

print("=" * 60)