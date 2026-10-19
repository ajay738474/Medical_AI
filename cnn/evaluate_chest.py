import os
import torch
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = (
    "cnn/models/"
    "chestmnist_resnet18_full_classifier.pth"
)

FEATURE_PATH = (
    "cnn/features/chest/"
    "test_features.pt"
)

LABEL_PATH = (
    "cnn/features/chest/"
    "test_labels.pt"
)

NUM_CLASSES = 14


# ============================================================
# 2. CHESTMNIST LABEL NAMES
# ============================================================

CLASS_NAMES = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltration",
    4: "Mass",
    5: "Nodule",
    6: "Pneumonia",
    7: "Pneumothorax",
    8: "Consolidation",
    9: "Edema",
    10: "Emphysema",
    11: "Fibrosis",
    12: "Pleural Thickening",
    13: "Hernia"
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CHESTMNIST EVALUATION")
print("=" * 70)

print("Device:", device)


# ============================================================
# 4. CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(
        f"Features not found: {FEATURE_PATH}"
    )

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(
        f"Labels not found: {LABEL_PATH}"
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

print("\nLoading trained ChestMNIST classifier...")

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

print("Classifier loaded.")

print(
    "Feature size:",
    feature_size
)

print(
    "Number of classes:",
    num_classes
)


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

print("\nRunning predictions...")

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

        probabilities = torch.sigmoid(
            outputs
        )

        all_probabilities.append(
            probabilities.cpu()
        )


probabilities = torch.cat(
    all_probabilities,
    dim=0
)

print(
    "Predictions completed:",
    probabilities.shape
)


# ============================================================
# 8. CONVERT TO NUMPY
# ============================================================

y_true = test_labels.numpy()

y_prob = probabilities.numpy()

y_pred = (
    y_prob >= 0.5
).astype(int)


# ============================================================
# 9. CALCULATE OVERALL METRICS
# ============================================================

print("\nCalculating metrics...")


try:

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        average="macro"
    )

except ValueError:

    macro_auc = float("nan")


try:

    macro_ap = average_precision_score(
        y_true,
        y_prob,
        average="macro"
    )

except ValueError:

    macro_ap = float("nan")


macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# 10. DISPLAY OVERALL RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CHESTMNIST TEST RESULTS")
print("=" * 70)

print(
    f"Test Images       : {len(y_true)}"
)

print(
    f"Macro AUROC       : {macro_auc * 100:.2f}%"
)

print(
    f"Macro Average Prec: {macro_ap * 100:.2f}%"
)

print(
    f"Macro Precision   : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall      : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1 Score    : {macro_f1 * 100:.2f}%"
)


# ============================================================
# 11. PER-CLASS METRICS
# ============================================================

print("\n")
print("=" * 70)
print("PER-LABEL RESULTS")
print("=" * 70)

print(
    f"{'ID':<4}"
    f"{'Finding':<25}"
    f"{'AUC':>10}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
)

print("-" * 70)


for class_id in range(
    NUM_CLASSES
):

    name = CLASS_NAMES.get(
        class_id,
        f"Class {class_id}"
    )

    true_class = y_true[
        :, class_id
    ]

    prob_class = y_prob[
        :, class_id
    ]

    pred_class = y_pred[
        :, class_id
    ]


    # AUC
    try:

        auc = roc_auc_score(
            true_class,
            prob_class
        )

    except ValueError:

        auc = float("nan")


    precision = precision_score(
        true_class,
        pred_class,
        zero_division=0
    )

    recall = recall_score(
        true_class,
        pred_class,
        zero_division=0
    )

    f1 = f1_score(
        true_class,
        pred_class,
        zero_division=0
    )


    print(
        f"{class_id:<4}"
        f"{name:<25}"
        f"{auc * 100:>9.2f}%"
        f"{precision * 100:>11.2f}%"
        f"{recall * 100:>9.2f}%"
        f"{f1 * 100:>9.2f}%"
    )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

RESULTS_DIR = "cnn/results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "chestmnist_evaluation.txt"
)


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CHESTMNIST EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Test Images: {len(y_true)}\n"
    )

    file.write(
        f"Macro AUROC: "
        f"{macro_auc * 100:.2f}%\n"
    )

    file.write(
        f"Macro Average Precision: "
        f"{macro_ap * 100:.2f}%\n"
    )

    file.write(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%\n"
    )

    file.write(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%\n"
    )

    file.write(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%\n"
    )

    file.write(
        "\nPER-LABEL RESULTS\n"
    )

    file.write(
        "=" * 70 + "\n"
    )

    for class_id in range(
        NUM_CLASSES
    ):

        name = CLASS_NAMES.get(
            class_id,
            f"Class {class_id}"
        )

        true_class = y_true[
            :, class_id
        ]

        prob_class = y_prob[
            :, class_id
        ]

        pred_class = y_pred[
            :, class_id
        ]

        try:

            auc = roc_auc_score(
                true_class,
                prob_class
            )

        except ValueError:

            auc = float("nan")

        precision = precision_score(
            true_class,
            pred_class,
            zero_division=0
        )

        recall = recall_score(
            true_class,
            pred_class,
            zero_division=0
        )

        f1 = f1_score(
            true_class,
            pred_class,
            zero_division=0
        )

        file.write(
            f"{class_id} - {name}: "
            f"AUC={auc * 100:.2f}%, "
            f"Precision={precision * 100:.2f}%, "
            f"Recall={recall * 100:.2f}%, "
            f"F1={f1 * 100:.2f}%\n"
        )


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