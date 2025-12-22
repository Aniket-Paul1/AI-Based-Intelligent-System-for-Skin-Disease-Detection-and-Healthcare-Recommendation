# ml_code/train_cnn.py

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight

from ml_code.config import (
    TRAIN_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    MODELS_DIR,
)

# -------------------------------
# Stability settings
# -------------------------------
tf.config.optimizer.set_jit(False)  # disable XLA for stability

# -------------------------------
# Hyperparameters
# -------------------------------
EPOCHS = 12
INITIAL_LR = 1e-4
FINE_TUNE_LR = 1e-5
FINE_TUNE_AT = 40  # unfreeze top N layers

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "cnn_model.h5"
CLASSES_PATH = MODELS_DIR / "classes.json"

# -------------------------------
# Data generators (with auto validation split)
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.15,
    shear_range=0.08,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = train_gen.num_classes
print("Classes:", train_gen.class_indices)

# Save class mapping
with open(CLASSES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)

# -------------------------------
# Compute class weights (CRITICAL)
# -------------------------------
class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = dict(enumerate(class_weights_arr))
print("Class weights:", class_weights)

# -------------------------------
# Build model
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

base_model.trainable = False  # freeze initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -------------------------------
# Callbacks
# -------------------------------
callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

# -------------------------------
# Steps (IMPORTANT FIX)
# -------------------------------
steps_per_epoch = train_gen.samples // train_gen.batch_size
validation_steps = val_gen.samples // val_gen.batch_size

# -------------------------------
# Phase 1: Train top classifier
# -------------------------------
print("\n=== Phase 1: Training classifier head ===\n")

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# -------------------------------
# Phase 2: Fine-tuning top layers
# -------------------------------
print("\n=== Phase 2: Fine-tuning CNN ===\n")

base_model.trainable = True
for layer in base_model.layers[:-FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_gen,
    validation_steps=validation_steps,
    epochs=6,
    callbacks=callbacks,
    class_weight=class_weights,
)

print("\nTraining complete.")
print("Best model saved to:", MODEL_PATH)
