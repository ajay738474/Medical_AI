import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

import medmnist
from medmnist import INFO

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

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

MODEL_PATH = (
    "cnn/models/"
    "retinamnist_resnet18_full_classifier.pth"
)

RESULT_DIR = "cnn/results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 64
IMAGE_SIZE = 128

DATA_FLAG = "retinamnist"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("RETINAMNIST MODEL EVALUATION")
print("=" * 70)

print(f"Device: {DEVICE}")
print()


# ============================================================
# DATASET INFORMATION
# ============================================================

info = INFO[DATA_FLAG]

print("Dataset information:")
print(f"Description : {info['description']}")
print(f"Task        : {info['task']}")
print(f"Channels    : {info['n_channels']}")
print(f"Classes     : {len(info['label'])}")
print()


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    int(k): v
    for k, v in info["label"].items()
}

print("Classes:")

for class_id, class_name in CLASS_NAMES.items():
    print(
        f"{class_id} - {class_name}"
    )

print()


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: Trained RetinaMNIST model not found.")
    print()
    print(
        f"Expected model: {MODEL_PATH}"
    )
    print()
    print(
        "Run cnn/train_retina.py first."
    )

    raise SystemExit


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([

    transforms.ToTensor(),

    # RetinaMNIST is RGB, so keep 3 channels.

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("Loading RetinaMNIST test dataset...")

DatasetClass = getattr(
    medmnist,
    info["python_class"]
)

test_dataset = DatasetClass(
    split="test",
    root=DATA_ROOT,
    transform=transform,
    download=False
)

print(
    f"Test images: {len(test_dataset)}"
)

print()


# ============================================================
# TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD RESNET-18
# ============================================================

print("Loading ResNet-18...")

resnet = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

for param in resnet.parameters():
    param.requires_grad = False

feature_size = resnet.fc.in_features

resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)

resnet.eval()

print("ResNet-18 loaded.")
print(
    f"Feature size: {feature_size}"
)

print()


# ============================================================
# LOAD CLASSIFIER
# ============================================================

num_classes = len(CLASS_NAMES)

classifier = nn.Linear(
    feature_size,
    num_classes
)

print(
    "Loading trained RetinaMNIST classifier..."
)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

classifier = classifier.to(DEVICE)

classifier.eval()

print("Classifier loaded.")
print()


# ============================================================
# EVALUATION
# ============================================================

print("=" * 70)
print("EVALUATING MODEL")
print("=" * 70)

all_predictions = []
all_labels = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        labels = (
            labels
            .view(-1)
            .long()
        )

        # Extract ResNet-18 features
        features = resnet(images)

        # Classifier prediction
        outputs = classifier(
            features
        )

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
    labels=list(CLASS_NAMES.keys()),
    target_names=list(CLASS_NAMES.values()),
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print("=" * 70)
print("RETINAMNIST EVALUATION RESULTS")
print("=" * 70)

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

print()

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)

print()

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

result_file = os.path.join(
    RESULT_DIR,
    "retinamnist_evaluation.txt"
)

with open(
    result_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RETINAMNIST MODEL EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Test Images : {len(all_labels)}\n"
    )

    f.write(
        f"Accuracy    : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision   : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall      : {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score    : {f1 * 100:.2f}%\n"
    )

    f.write("\n")

    f.write(
        "CLASSIFICATION REPORT\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(report)

    f.write("\n\n")

    f.write(
        "CONFUSION MATRIX\n"
    )

    f.write(
        "=" * 70 + "\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# COMPLETE
# ============================================================

print()

print("=" * 70)
print("RETINAMNIST EVALUATION COMPLETED")
print("=" * 70)

print(
    f"Results saved to: {result_file}"
)

print()

print("Next step:")
print("Create cnn/predict_retina.py")

print("=" * 70)