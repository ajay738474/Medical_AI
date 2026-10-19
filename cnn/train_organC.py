import os
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from torchvision import transforms

from medmnist import OrganCMNIST


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 64

NUM_CLASSES = 11

BATCH_SIZE = 64

EPOCHS = 3

TRAIN_PER_CLASS = 500

VAL_PER_CLASS = 100

LEARNING_RATE = 0.001

MODEL_PATH = (
    "cnn/models/"
    "organcmnist_fast_cnn_classifier.pth"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    "cnn/models",
    exist_ok=True
)

os.makedirs(
    "cnn/results",
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# INFORMATION
# ============================================================

print("=" * 70)
print("OrganCMNIST - Fast CNN Training")
print("=" * 70)

print("Device:", device)
print("Image size:", IMAGE_SIZE)
print("Classes:", NUM_CLASSES)
print("Train images per class:", TRAIN_PER_CLASS)
print("Validation images per class:", VAL_PER_CLASS)
print("Epochs:", EPOCHS)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(42)

np.random.seed(42)

torch.manual_seed(42)


# ============================================================
# IMAGE TRANSFORM
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
# LOAD DATASET
# ============================================================

print("\nLoading OrganCMNIST...")

train_full = OrganCMNIST(
    split="train",
    transform=transform,
    download=True
)

val_full = OrganCMNIST(
    split="val",
    transform=transform,
    download=True
)

test_full = OrganCMNIST(
    split="test",
    transform=transform,
    download=True
)


print("\nDataset sizes:")

print(
    "Full train:",
    len(train_full)
)

print(
    "Full validation:",
    len(val_full)
)

print(
    "Full test:",
    len(test_full)
)


# ============================================================
# GET LABELS
# ============================================================

train_labels = np.array(
    train_full.labels
).reshape(-1)

val_labels = np.array(
    val_full.labels
).reshape(-1)


# ============================================================
# CREATE BALANCED SUBSET
# ============================================================

def create_balanced_indices(
    labels,
    samples_per_class,
    number_of_classes
):

    selected_indices = []

    print()

    for class_id in range(
        number_of_classes
    ):

        class_indices = np.where(
            labels == class_id
        )[0]

        np.random.shuffle(
            class_indices
        )

        selected = class_indices[
            :samples_per_class
        ]

        selected_indices.extend(
            selected.tolist()
        )

        print(
            f"Class {class_id}: "
            f"{len(selected)} images"
        )

    np.random.shuffle(
        selected_indices
    )

    return selected_indices


# ============================================================
# TRAIN SUBSET
# ============================================================

print(
    "\nCreating balanced training subset..."
)

train_indices = create_balanced_indices(
    train_labels,
    TRAIN_PER_CLASS,
    NUM_CLASSES
)


# ============================================================
# VALIDATION SUBSET
# ============================================================

print(
    "\nCreating balanced validation subset..."
)

val_indices = create_balanced_indices(
    val_labels,
    VAL_PER_CLASS,
    NUM_CLASSES
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = Subset(
    train_full,
    train_indices
)

val_dataset = Subset(
    val_full,
    val_indices
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nSelected training images:")

print(
    len(train_dataset)
)

print(
    "Selected validation images:",
    len(val_dataset)
)


# ============================================================
# LIGHTWEIGHT CNN
# ============================================================

class OrganCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
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

print("\nCreating lightweight CNN...")

model = OrganCNN()

model = model.to(device)

print(model)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


print("\nStarting training...")

print("=" * 70)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = (
            labels
            .squeeze()
            .long()
            .to(device)
        )


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_total += (
            labels.size(0)
        )


        train_correct += (
            predictions == labels
        ).sum().item()


    # --------------------------------------------------------
    # TRAIN METRICS
    # --------------------------------------------------------

    train_loss = (
        running_loss
        / len(train_loader)
    )


    train_accuracy = (
        100.0
        * train_correct
        / train_total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

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


            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_total += (
                labels.size(0)
            )


            val_correct += (
                predictions == labels
            ).sum().item()


    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    val_loss = (
        val_loss_total
        / len(val_loader)
    )


    val_accuracy = (
        100.0
        * val_correct
        / val_total
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "Best model saved:",
            MODEL_PATH
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print("=" * 70)

print("Training completed.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Model saved at:"
)

print(
    MODEL_PATH
)

print("=" * 70)