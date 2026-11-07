import os
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from pathlib import Path

from torch.utils.data import (
    DataLoader,
    random_split
)

from torchvision import (
    datasets,
    transforms
)

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from classifier.model import HazeCNN


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 128

BATCH_SIZE = 32

MAX_EPOCHS = 40

INITIAL_LR = 1e-3

VALIDATION_RATIO = 0.20

EARLY_STOPPING_PATIENCE = 7

MIN_DELTA = 1e-4


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent


DATASET_DIR = (

    BASE_DIR
    /
    "dataset"
    /
    "CNN_Dataset"

)


TRAIN_DIR = (

    DATASET_DIR
    /
    "train"

)


TEST_DIR = (

    DATASET_DIR
    /
    "test"

)


MODEL_DIR = (

    BASE_DIR
    /
    "models"

)


RESULT_DIR = (

    BASE_DIR
    /
    "results"

)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

MODEL_DIR.mkdir(

    parents=True,

    exist_ok=True

)


RESULT_DIR.mkdir(

    parents=True,

    exist_ok=True

)


# ============================================================
# SET RANDOM SEED
# ============================================================

random.seed(
    SEED
)


np.random.seed(
    SEED
)


torch.manual_seed(
    SEED
)


if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        SEED
    )


# ============================================================
# DEVICE
# ============================================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)


print("\n============================================")

print(" HAZY / CLEAR CNN TRAINING")

print("============================================")

print(
    f"\nDevice: {device}"
)


# ============================================================
# CHECK DATASET
# ============================================================

if not TRAIN_DIR.exists():

    raise FileNotFoundError(

        f"\nTraining folder not found:\n{TRAIN_DIR}"

    )


if not TEST_DIR.exists():

    raise FileNotFoundError(

        f"\nTesting folder not found:\n{TEST_DIR}"

    )


print(
    "\nDataset found successfully."
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([


    transforms.Resize(

        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )

    ),


    transforms.ToTensor()

])


# ============================================================
# LOAD DATASET
# ============================================================

full_train_dataset = datasets.ImageFolder(

    root=str(
        TRAIN_DIR
    ),

    transform=transform

)


test_dataset = datasets.ImageFolder(

    root=str(
        TEST_DIR
    ),

    transform=transform

)


# ============================================================
# DISPLAY CLASSES
# ============================================================

print("\n============================================")

print(" DATASET INFORMATION")

print("============================================")


print(

    f"\nClasses found: "

    f"{full_train_dataset.classes}"

)


print(

    f"Class mapping: "

    f"{full_train_dataset.class_to_idx}"

)


print(

    f"\nTotal training images: "

    f"{len(full_train_dataset)}"

)


print(

    f"Total testing images: "

    f"{len(test_dataset)}"

)


# ============================================================
# VERIFY EXPECTED CLASSES
# ============================================================

expected_classes = {

    "clear",

    "hazy"

}


if set(
    full_train_dataset.classes
) != expected_classes:

    raise ValueError(

        "\nDataset must contain exactly:\n"

        "clear/\n"

        "hazy/\n"

    )


# ============================================================
# SPLIT TRAINING INTO
#
# TRAINING + VALIDATION
# ============================================================

num_total = len(
    full_train_dataset
)


num_validation = max(

    1,

    int(
        VALIDATION_RATIO
        *
        num_total
    )

)


num_training = (

    num_total

    -

    num_validation

)


generator = torch.Generator().manual_seed(

    SEED

)


train_dataset, val_dataset = random_split(

    full_train_dataset,

    [

        num_training,

        num_validation

    ],

    generator=generator

)


print("\nDataset split:")

print(
    f"Training   : {len(train_dataset)}"
)

print(
    f"Validation : {len(val_dataset)}"
)

print(
    f"Test       : {len(test_dataset)}"
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


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0

)


# ============================================================
# CREATE CNN
# ============================================================

model = HazeCNN().to(

    device

)


print("\n============================================")

print(" CNN ARCHITECTURE")

print("============================================\n")


print(
    model
)


# ============================================================
# LOSS FUNCTION
#
# Binary Classification
# ============================================================

criterion = nn.BCELoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(

    model.parameters(),

    lr=INITIAL_LR

)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3,

    min_lr=1e-6

)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "epoch": [],

    "train_loss": [],

    "train_accuracy": [],

    "val_loss": [],

    "val_accuracy": [],

    "learning_rate": []

}


best_val_loss = float(
    "inf"
)


best_model_state = None


epochs_without_improvement = 0


# ============================================================
# START TRAINING
# ============================================================

print("\n============================================")

print(" STARTING CNN TRAINING")

print("============================================\n")


for epoch in range(

    1,

    MAX_EPOCHS + 1

):


    # ========================================================
    # TRAINING MODE
    # ========================================================

    model.train()


    running_loss = 0.0


    correct = 0


    total = 0


    progress = tqdm(

        train_loader,

        desc=f"Epoch {epoch}/{MAX_EPOCHS}"

    )


    for images, labels in progress:


        images = images.to(

            device

        )


        labels = labels.float().to(

            device

        ).unsqueeze(

            1

        )


        # ----------------------------------------------------
        # RESET GRADIENT
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD PASS
        # ----------------------------------------------------

        outputs = model(

            images

        )


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        loss = criterion(

            outputs,

            labels

        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        optimizer.step()


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        running_loss += (

            loss.item()

            *

            images.size(0)

        )


        predictions = (

            outputs >= 0.5

        ).float()


        correct += (

            predictions == labels

        ).sum().item()


        total += (

            labels.size(0)

        )


        progress.set_postfix(

            loss=f"{loss.item():.4f}"

        )


    train_loss = (

        running_loss

        /

        total

    )


    train_accuracy = (

        correct

        /

        total

    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()


    val_loss_total = 0.0


    val_correct = 0


    val_total = 0


    with torch.no_grad():


        for images, labels in val_loader:


            images = images.to(

                device

            )


            labels = labels.float().to(

                device

            ).unsqueeze(

                1

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

                *

                images.size(0)

            )


            predictions = (

                outputs >= 0.5

            ).float()


            val_correct += (

                predictions == labels

            ).sum().item()


            val_total += (

                labels.size(0)

            )


    val_loss = (

        val_loss_total

        /

        val_total

    )


    val_accuracy = (

        val_correct

        /

        val_total

    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(

        val_loss

    )


    current_lr = (

        optimizer.param_groups[0]["lr"]

    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history["epoch"].append(

        epoch

    )


    history["train_loss"].append(

        train_loss

    )


    history["train_accuracy"].append(

        train_accuracy

    )


    history["val_loss"].append(

        val_loss

    )


    history["val_accuracy"].append(

        val_accuracy

    )


    history["learning_rate"].append(

        current_lr

    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(

        f"\nEpoch {epoch}/{MAX_EPOCHS}"

    )


    print(

        f"Train Loss: "

        f"{train_loss:.6f}"

    )


    print(

        f"Train Accuracy: "

        f"{train_accuracy * 100:.2f}%"

    )


    print(

        f"Validation Loss: "

        f"{val_loss:.6f}"

    )


    print(

        f"Validation Accuracy: "

        f"{val_accuracy * 100:.2f}%"

    )


    print(

        f"Learning Rate: "

        f"{current_lr:.8f}"

    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < (

        best_val_loss

        -

        MIN_DELTA

    ):


        best_val_loss = (

            val_loss

        )


        best_model_state = copy.deepcopy(

            model.state_dict()

        )


        epochs_without_improvement = 0


        checkpoint = {


            "model_state_dict":

                model.state_dict(),


            "epoch":

                epoch,


            "val_loss":

                val_loss,


            "val_accuracy":

                val_accuracy,


            "class_to_idx":

                full_train_dataset.class_to_idx,


            "image_size":

                IMAGE_SIZE

        }


        torch.save(

            checkpoint,

            MODEL_DIR
            /
            "best_haze_CNN.pth"

        )


        print(

            "\n✓ New best model saved."

        )


    else:


        epochs_without_improvement += 1


        print(

            f"No significant improvement "

            f"({epochs_without_improvement}/"

            f"{EARLY_STOPPING_PATIENCE})"

        )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if (

        epochs_without_improvement

        >=

        EARLY_STOPPING_PATIENCE

    ):


        print(

            "\nEarly stopping triggered."

        )


        break


# ============================================================
# RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:


    model.load_state_dict(

        best_model_state

    )


    print(

        "\nBest validation model restored."

    )


# ============================================================
# SAVE FINAL MODEL
# ============================================================

torch.save(

    model.state_dict(),

    MODEL_DIR
    /
    "haze_CNN_final.pth"

)


# ============================================================
# PLOT LOSS
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    history["epoch"],

    history["train_loss"],

    label="Training Loss"

)


plt.plot(

    history["epoch"],

    history["val_loss"],

    label="Validation Loss"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Loss"

)


plt.title(

    "CNN Training and Validation Loss"

)


plt.legend()


plt.grid(

    True

)


plt.tight_layout()


plt.savefig(

    RESULT_DIR
    /
    "training_loss.png"

)


plt.close()


# ============================================================
# PLOT ACCURACY
# ============================================================

plt.figure(

    figsize=(10, 5)

)


plt.plot(

    history["epoch"],

    np.array(
        history["train_accuracy"]
    )
    *
    100,

    label="Training Accuracy"

)


plt.plot(

    history["epoch"],

    np.array(
        history["val_accuracy"]
    )
    *
    100,

    label="Validation Accuracy"

)


plt.xlabel(

    "Epoch"

)


plt.ylabel(

    "Accuracy (%)"

)


plt.title(

    "CNN Training and Validation Accuracy"

)


plt.legend()


plt.grid(

    True

)


plt.tight_layout()


plt.savefig(

    RESULT_DIR
    /
    "training_accuracy.png"

)


plt.close()


# ============================================================
# FINAL TESTING
# ============================================================

print("\n============================================")

print(" FINAL TESTING")

print("============================================")


model.eval()


all_scores = []


all_labels = []


with torch.no_grad():


    for images, labels in test_loader:


        images = images.to(

            device

        )


        outputs = model(

            images

        )


        scores = (

            outputs

            .cpu()

            .numpy()

            .flatten()

        )


        all_scores.extend(

            scores

        )


        all_labels.extend(

            labels.numpy()

        )


all_scores = np.array(

    all_scores

)


all_labels = np.array(

    all_labels

)


# ============================================================
# BINARY CLASSIFICATION
# ============================================================

predictions = (

    all_scores >= 0.5

).astype(

    int

)


test_accuracy = accuracy_score(

    all_labels,

    predictions

)


print(

    f"\nTest Accuracy: "

    f"{test_accuracy * 100:.2f}%"

)


print(

    f"\nCorrect predictions: "

    f"{np.sum(predictions == all_labels)}"

)


print(

    f"Total test images: "

    f"{len(all_labels)}"

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    all_labels,

    predictions

)


print(

    "\nConfusion Matrix:"

)


print(

    cm

)


print(

    "\nClassification Report:\n"

)


print(

    classification_report(

        all_labels,

        predictions,

        target_names=[

            "clear",

            "hazy"

        ]

    )

)


# ============================================================
# SAVE RESULTS
# ============================================================

np.savetxt(

    RESULT_DIR
    /
    "confusion_matrix.csv",

    cm,

    delimiter=",",

    fmt="%d"

)


print("\n============================================")

print(" CNN TRAINING COMPLETED")

print("============================================")


print(

    "\nBest model:"

)


print(

    MODEL_DIR
    /
    "best_haze_CNN.pth"

)


print(

    "\nFinal model:"

)


print(

    MODEL_DIR
    /
    "haze_CNN_final.pth"

)