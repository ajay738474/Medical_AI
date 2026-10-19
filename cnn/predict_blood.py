import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import BloodMNIST
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

BATCH_SIZE = 128
IMAGE_SIZE = 128
NUM_CLASSES = 8

MODEL_PATH = "cnn/models/bloodmnist_resnet18_full_classifier.pth"
RESULT_PATH = "cnn/results/bloodmnist_evaluation.txt"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BloodMNIST - Model Evaluation")
print("=" * 60)

print("Device:", device)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading BloodMNIST test dataset...")

test_dataset = BloodMNIST(
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
# LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

model = models.resnet18(weights=None)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

all_predictions = []
all_labels = []

print("\nRunning evaluation...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.squeeze().long().to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
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

class_names = [
    "Basophils",
    "Eosinophils",
    "Erythroblasts",
    "Immature granulocytes",
    "Lymphocytes",
    "Monocytes",
    "Neutrophils",
    "Platelets"
]

report = classification_report(
    all_labels,
    all_predictions,
    target_names=class_names,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BloodMNIST Evaluation Results")
print("=" * 60)

print(
    f"Test Images : {len(all_labels)}"
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
print(matrix)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "BloodMNIST Evaluation Results\n"
    )

    file.write("=" * 60 + "\n")

    file.write(
        f"Test Images : {len(all_labels)}\n"
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
        f"F1 Score    : {f1 * 100:.2f}%\n"
    )

    file.write(
        "\nClassification Report:\n"
    )

    file.write(report)

    file.write(
        "\nConfusion Matrix:\n"
    )

    file.write(
        str(matrix)
    )


print("\nResults saved to:")
print(RESULT_PATH)

print("=" * 60)