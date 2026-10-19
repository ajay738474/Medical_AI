import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import OrganSMNIST, INFO

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# CONFIG
# =========================

IMAGE_SIZE = 64
BATCH_SIZE = 64
NUM_CLASSES = 11

MODEL_PATH = "cnn/models/organsmnist_fast_cnn_classifier.pth"
RESULT_PATH = "cnn/results/organsmnist_evaluation.txt"


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================
# CLASS NAMES
# =========================

info = INFO["organsmnist"]

CLASS_NAMES = {
    int(k): v
    for k, v in info["label"].items()
}

print("\nClasses:")
for index, name in CLASS_NAMES.items():
    print(index, ":", name)


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# =========================
# LOAD TEST DATA
# =========================

print("\nLoading OrganSMNIST test dataset...")

test_dataset = OrganSMNIST(
    split="test",
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))


# =========================
# MODEL
# =========================

class OrganCNN(nn.Module):

    def __init__(self, num_classes=11):

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

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================
# LOAD MODEL
# =========================

print("\nLoading trained model...")

model = OrganCNN(NUM_CLASSES)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully.")


# =========================
# EVALUATION
# =========================

all_predictions = []
all_labels = []

print("\nEvaluating OrganSMNIST...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.squeeze().long()

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        ).cpu()

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# =========================
# METRICS
# =========================

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


# =========================
# PRINT RESULTS
# =========================

print("\n===================================")
print("      OrganSMNIST Evaluation")
print("===================================")

print(f"Test Images      : {len(test_dataset)}")
print(f"Accuracy         : {accuracy * 100:.2f}%")
print(f"Macro Precision  : {precision * 100:.2f}%")
print(f"Macro Recall     : {recall * 100:.2f}%")
print(f"Macro F1 Score   : {f1 * 100:.2f}%")


# =========================
# CLASSIFICATION REPORT
# =========================

target_names = [
    CLASS_NAMES[i]
    for i in range(NUM_CLASSES)
]

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")

print(cm)


# =========================
# SAVE RESULTS
# =========================

os.makedirs(
    "cnn/results",
    exist_ok=True
)

with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("OrganSMNIST Evaluation\n")
    f.write("======================\n\n")

    f.write(
        f"Test Images      : {len(test_dataset)}\n"
    )

    f.write(
        f"Accuracy         : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Macro Precision  : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Macro Recall     : {recall * 100:.2f}%\n"
    )

    f.write(
        f"Macro F1 Score   : {f1 * 100:.2f}%\n\n"
    )

    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)

    f.write("\nConfusion Matrix\n")
    f.write("================\n")
    f.write(str(cm))


print("\nEvaluation saved to:")
print(RESULT_PATH)