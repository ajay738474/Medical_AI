import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import medmnist
from medmnist import INFO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = r"C:\Users\ajay1\.medmnist"

MODEL_DIR = "cnn/models"
FEATURE_DIR = "cnn/features/oct"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 5
IMAGE_SIZE = 128


print("=" * 70)
print("OCTMNIST RESNET-18 TRAINING")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print()


# ============================================================
# LOAD OCTMNIST INFORMATION
# ============================================================

DATA_FLAG = "octmnist"

info = INFO[DATA_FLAG]

print("Dataset information:")
print(f"Name       : {info['description']}")
print(f"Task       : {info['task']}")
print(f"Channels   : {info['n_channels']}")
print(f"Classes    : {len(info['label'])}")
print()


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),

    # OCTMNIST images are grayscale.
    # Convert 1 channel -> 3 channels for ImageNet ResNet-18.
    transforms.Lambda(
        lambda x: x.repeat(3, 1, 1)
        if x.shape[0] == 1
        else x
    ),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading OCTMNIST datasets...")

DatasetClass = getattr(medmnist, info["python_class"])

train_dataset = DatasetClass(
    split="train",
    root=DATA_ROOT,
    transform=transform,
    download=False
)

val_dataset = DatasetClass(
    split="val",
    root=DATA_ROOT,
    transform=transform,
    download=False
)

test_dataset = DatasetClass(
    split="test",
    root=DATA_ROOT,
    transform=transform,
    download=False
)

print()
print("Dataset loaded successfully.")

print(f"Train images : {len(train_dataset)}")
print(f"Val images   : {len(val_dataset)}")
print(f"Test images  : {len(test_dataset)}")
print()


# ============================================================
# DATALOADERS
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

print("Loading pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT

resnet = resnet18(weights=weights)

# Freeze pretrained layers
for param in resnet.parameters():
    param.requires_grad = False


# ============================================================
# EXTRACT FEATURE SIZE
# ============================================================

feature_size = resnet.fc.in_features

# Remove original ImageNet classifier
resnet.fc = nn.Identity()

resnet = resnet.to(DEVICE)

print("ResNet-18 loaded.")
print(f"Feature size: {feature_size}")
print()


# ============================================================
# DETERMINE NUMBER OF CLASSES
# ============================================================

num_classes = len(info["label"])

print(f"Number of classes: {num_classes}")
print()

# OCTMNIST is a multi-class classification task.
classifier = nn.Linear(feature_size, num_classes)

classifier = classifier.to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(loader, split_name):
    """
    Extract ResNet-18 features and cache them.
    """

    feature_file = os.path.join(
        FEATURE_DIR,
        f"{split_name}_features.pt"
    )

    label_file = os.path.join(
        FEATURE_DIR,
        f"{split_name}_labels.pt"
    )

    # Use cached features if available
    if os.path.exists(feature_file) and os.path.exists(label_file):

        print(f"Loading cached {split_name} features...")

        features = torch.load(
            feature_file,
            map_location="cpu",
            weights_only=False
        )

        labels = torch.load(
            label_file,
            map_location="cpu",
            weights_only=False
        )

        return features, labels

    print(f"Extracting {split_name} features...")

    resnet.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            features = resnet(images)

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.view(-1).long().cpu()
            )

    features = torch.cat(all_features)
    labels = torch.cat(all_labels)

    torch.save(features, feature_file)
    torch.save(labels, label_file)

    print(
        f"{split_name} features saved: "
        f"{features.shape}"
    )

    return features, labels


# ============================================================
# EXTRACT FEATURES
# ============================================================

train_features, train_labels = extract_features(
    train_loader,
    "train"
)

val_features, val_labels = extract_features(
    val_loader,
    "val"
)

test_features, test_labels = extract_features(
    test_loader,
    "test"
)

print()


# ============================================================
# CREATE FEATURE DATALOADERS
# ============================================================

train_feature_dataset = torch.utils.data.TensorDataset(
    train_features,
    train_labels
)

val_feature_dataset = torch.utils.data.TensorDataset(
    val_features,
    val_labels
)

train_feature_loader = DataLoader(
    train_feature_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_feature_loader = DataLoader(
    val_feature_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# TRAIN CLASSIFIER
# ============================================================

print("=" * 70)
print("TRAINING CLASSIFIER")
print("=" * 70)

best_val_accuracy = 0.0

model_path = os.path.join(
    MODEL_DIR,
    "octmnist_resnet18_full_classifier.pth"
)


for epoch in range(EPOCHS):

    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_feature_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = classifier(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    classifier.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for features, labels in val_feature_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = classifier(features)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = 100 * val_correct / val_total


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}%"
    )


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            classifier.state_dict(),
            model_path
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Acc: {val_accuracy:.2f}%)"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print()
print("=" * 70)
print("OCTMNIST TRAINING COMPLETED")
print("=" * 70)

print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Model saved to: {model_path}")

print()
print("Next step:")
print("Create cnn/evaluate_oct.py")
print("=" * 70)