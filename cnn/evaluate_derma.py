import os
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = (
    "cnn/models/"
    "dermamnist_resnet18_full_classifier.pth"
)

FEATURE_PATH = (
    "cnn/features/derma/"
    "test_features.pt"
)

LABEL_PATH = (
    "cnn/features/derma/"
    "test_labels.pt"
)

NUM_CLASSES = 7


# ============================================================
# 2. DERMAMNIST CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions"
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DERMAMNIST EVALUATION")
print("=" * 70)

print(
    "Device:",
    device
)


# ============================================================
# 4. CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(FEATURE_PATH):

    raise FileNotFoundError(
        f"Test features not found: {FEATURE_PATH}"
    )

if not os.path.exists(LABEL_PATH):

    raise FileNotFoundError(
        f"Test labels not found: {LABEL_PATH}"
    )


# ============================================================
# 5. LOAD TEST FEATURES
# ============================================================

print("\nLoading test features...")

test_features = torch.load(
    FEATURE_PATH,
    map_location="cpu",
    weights_only=True
)

test_labels = torch.load(
    LABEL_PATH,
    map_location="cpu",
    weights_only=True
)

print(
    "Test features:",
    test_features.shape
)

print(
    "Test labels:",
    test_labels.shape
)


# ============================================================
# 6. LOAD TRAINED CLASSIFIER
# ============================================================

print(
    "\nLoading trained DermaMNIST classifier..."
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

feature_size = checkpoint[
    "feature_size"
]

num_classes = checkpoint[
    "class_count"
]

classifier = torch.nn.Linear(
    feature_size,
    num_classes
)

classifier.load_state_dict(
    checkpoint[
        "classifier_state_dict"
    ]
)

classifier = classifier.to(
    device
)

classifier.eval()

print(
    "Classifier loaded."
)

print(
    "Feature size:",
    feature_size
)

print(
    "Number of classes:",
    num_classes
)


# ============================================================
# 7. RUN TEST PREDICTIONS
# ============================================================

print(
    "\nRunning predictions..."
)

all_predictions = []

all_probabilities = []

batch_size = 1024


with torch.no_grad():

    for start in range(
        0,
        len(test_features),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(test_features)
        )

        features = test_features[
            start:end
        ].to(device)

        outputs = classifier(
            features
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_predictions.append(
            predictions.cpu()
        )

        all_probabilities.append(
            probabilities.cpu()
        )


predictions = torch.cat(
    all_predictions
).numpy()

probabilities = torch.cat(
    all_probabilities
).numpy()

true_labels = test_labels.numpy()

true_labels = true_labels.reshape(-1)


print(
    "Predictions completed:",
    len(predictions)
)


# ============================================================
# 8. OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

precision = precision_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# 9. DISPLAY OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DERMAMNIST TEST RESULTS")
print("=" * 70)

print(
    f"Test Images : {len(true_labels)}"
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


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

target_names = [
    CLASS_NAMES[i]
    for i in range(NUM_CLASSES)
]

report = classification_report(
    true_labels,
    predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=list(range(NUM_CLASSES))
)

print(
    "\nRows = Actual"
)

print(
    "Columns = Predicted\n"
)

print(cm)


# ============================================================
# 12. PER-CLASS ACCURACY
# ============================================================

print("\n")
print("=" * 70)
print("PER-CLASS RESULTS")
print("=" * 70)

for class_id in range(
    NUM_CLASSES
):

    actual_count = np.sum(
        true_labels == class_id
    )

    correct_count = np.sum(
        (
            true_labels == class_id
        )
        &
        (
            predictions == class_id
        )
    )

    if actual_count > 0:

        class_accuracy = (
            correct_count
            / actual_count
            * 100
        )

    else:

        class_accuracy = 0.0


    print(
        f"{class_id} - "
        f"{CLASS_NAMES[class_id]:<30}"
        f"Accuracy: "
        f"{class_accuracy:.2f}% "
        f"({correct_count}/{actual_count})"
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

RESULTS_DIR = "cnn/results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "dermamnist_evaluation.txt"
)


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "DERMAMNIST EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Test Images : "
        f"{len(true_labels)}\n"
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
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        report
    )

    file.write(
        "\n\nCONFUSION MATRIX\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    file.write(
        str(cm)
    )

    file.write(
        "\n\nPER-CLASS RESULTS\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        actual_count = np.sum(
            true_labels == class_id
        )

        correct_count = np.sum(
            (
                true_labels == class_id
            )
            &
            (
                predictions == class_id
            )
        )

        if actual_count > 0:

            class_accuracy = (
                correct_count
                / actual_count
                * 100
            )

        else:

            class_accuracy = 0.0


        file.write(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]}: "
            f"{class_accuracy:.2f}% "
            f"({correct_count}/"
            f"{actual_count})\n"
        )


# ============================================================
# 14. COMPLETED
# ============================================================

print("\n")
print("=" * 70)

print(
    "Evaluation completed."
)

print(
    "Results saved:"
)

print(
    RESULT_FILE
)

print("=" * 70)