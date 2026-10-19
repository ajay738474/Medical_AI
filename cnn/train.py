import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import PathMNIST


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# 2. SETTINGS
# ============================================================

BATCH_SIZE = 64
FEATURE_BATCH_SIZE = 512
NUM_CLASSES = 9
EPOCHS = 5

FEATURE_DIR = "cnn/features"
MODEL_DIR = "cnn/models"


# ============================================================
# 3. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. LOAD FULL PATHMNIST
# ============================================================

print("\nLoading PathMNIST...")

train_dataset = PathMNIST(
    split="train",
    transform=transform,
    download=True
)

val_dataset = PathMNIST(
    split="val",
    transform=transform,
    download=True
)

test_dataset = PathMNIST(
    split="test",
    transform=transform,
    download=True
)

print("Training images  :", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images   :", len(test_dataset))


# ============================================================
# 5. DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 6. LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()

print("ResNet-18 loaded and frozen.")


# ============================================================
# 7. CREATE FEATURE DIRECTORY
# ============================================================

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 8. FEATURE EXTRACTION
# ============================================================

def extract_features(loader, name):

    features_file = os.path.join(
        FEATURE_DIR,
        f"{name}_features.pt"
    )

    labels_file = os.path.join(
        FEATURE_DIR,
        f"{name}_labels.pt"
    )

    # If already extracted, load them
    if os.path.exists(features_file) and os.path.exists(labels_file):

        print(
            f"\n{name} features already exist."
        )

        features = torch.load(
            features_file,
            weights_only=True
        )

        labels = torch.load(
            labels_file,
            weights_only=True
        )

        print(
            f"Loaded {name} features:",
            features.shape
        )

        return features, labels


    print("\nExtracting", name, "features...")

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_number, (images, labels) in enumerate(loader):

            images = images.to(device)

            features = feature_extractor(images)

            features = features.view(
                features.size(0),
                -1
            )

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels.squeeze().long()
            )

            if (batch_number + 1) % 50 == 0:

                processed = (
                    (batch_number + 1)
                    * loader.batch_size
                )

                print(
                    f"{name}: processed approximately "
                    f"{processed} images"
                )


    features = torch.cat(all_features)

    labels = torch.cat(all_labels)


    torch.save(
        features,
        features_file
    )

    torch.save(
        labels,
        labels_file
    )

    print(
        f"{name} features saved:",
        features.shape
    )

    return features, labels


# ============================================================
# 9. EXTRACT TRAIN FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TRAIN FEATURE EXTRACTION")
print("=" * 60)

train_features, train_labels = extract_features(
    train_loader,
    "train"
)


# ============================================================
# 10. EXTRACT VALIDATION FEATURES
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION FEATURE EXTRACTION")
print("=" * 60)

val_features, val_labels = extract_features(
    val_loader,
    "val"
)


# ============================================================
# 11. EXTRACT TEST FEATURES
# ============================================================

print("\n" + "=" * 60)
print("TEST FEATURE EXTRACTION")
print("=" * 60)

test_features, test_labels = extract_features(
    test_loader,
    "test"
)


# ============================================================
# 12. FEATURE DATALOADERS
# ============================================================

train_feature_loader = DataLoader(
    TensorDataset(
        train_features,
        train_labels
    ),
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=True
)

val_feature_loader = DataLoader(
    TensorDataset(
        val_features,
        val_labels
    ),
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=False
)

test_feature_loader = DataLoader(
    TensorDataset(
        test_features,
        test_labels
    ),
    batch_size=FEATURE_BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 13. CLASSIFIER
# ============================================================

classifier = torch.nn.Linear(
    512,
    NUM_CLASSES
)

classifier = classifier.to(device)

print("\nClassifier created.")

print("Input features :", 512)
print("Output classes :", NUM_CLASSES)


# ============================================================
# 14. LOSS + OPTIMIZER
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=0.001
)


# ============================================================
# 15. TRAIN CLASSIFIER
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFIER TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_feature_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(features)

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

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss /
        len(train_feature_loader)
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    classifier.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for features, labels in val_feature_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = classifier(features)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Training Loss       : {train_loss:.4f}"
    )

    print(
        f"Training Accuracy   : {train_accuracy:.2f}%"
    )

    print(
        f"Validation Accuracy : {val_accuracy:.2f}%"
    )


# ============================================================
# 16. FINAL TEST
# ============================================================

classifier.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for features, labels in test_feature_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = classifier(features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        test_total += labels.size(0)

        test_correct += (
            predictions == labels
        ).sum().item()


test_accuracy = (
    100 * test_correct / test_total
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "pathmnist_resnet18_full_classifier.pth"
)

torch.save(
    {
        "classifier_state_dict":
            classifier.state_dict(),

        "class_count":
            NUM_CLASSES,

        "feature_size":
            512,

        "test_accuracy":
            test_accuracy
    },
    model_path
)


# ============================================================
# 18. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FULL PATHMNIST TRAINING COMPLETED")
print("=" * 60)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)

print(
    f"Final Test Accuracy: {test_accuracy:.2f}%"
)

print(
    "Model saved:",
    model_path
)

print("=" * 60)