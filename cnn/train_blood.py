import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from medmnist import BloodMNIST


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
IMAGE_SIZE = 128
NUM_CLASSES = 8
LEARNING_RATE = 0.001

MODEL_DIR = "cnn/models"
MODEL_PATH = "cnn/models/bloodmnist_resnet18_full_classifier.pth"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("BloodMNIST - ResNet18 Training")
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
# LOAD BLOODMNIST DATASET
# ============================================================

print("\nLoading BloodMNIST dataset...")

train_dataset = BloodMNIST(
    split="train",
    transform=transform,
    download=True
)

val_dataset = BloodMNIST(
    split="val",
    transform=transform,
    download=True
)

test_dataset = BloodMNIST(
    split="test",
    transform=transform,
    download=True
)

print("Train images :", len(train_dataset))
print("Validation   :", len(val_dataset))
print("Test images  :", len(test_dataset))


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


# ============================================================
# LOAD PRETRAINED RESNET18
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


# ============================================================
# FREEZE PRETRAINED LAYERS
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    NUM_CLASSES
)

# Only classifier will train
for parameter in model.fc.parameters():
    parameter.requires_grad = True

model = model.to(device)


print("ResNet18 loaded.")
print("Number of classes:", NUM_CLASSES)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

print("\nStarting training...")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)

        # BloodMNIST labels can have shape [batch, 1]
        labels = labels.squeeze().long().to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.squeeze().long().to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            validation_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            validation_total += labels.size(0)

            validation_correct += (
                predictions == labels
            ).sum().item()

    val_loss = (
        validation_loss /
        len(val_loader)
    )

    val_accuracy = (
        100.0 *
        validation_correct /
        validation_total
    )


    # --------------------------------------------------------
    # PRINT RESULTS
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
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("Training completed.")
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)
print(
    "Model saved at:",
    MODEL_PATH
)
print("=" * 60)