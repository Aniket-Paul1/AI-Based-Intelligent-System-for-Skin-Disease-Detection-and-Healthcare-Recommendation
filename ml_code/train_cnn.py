# train_cnn.py
import json
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ml_code.config import DATA_DIR, MODELS_DIR, IMG_SIZE, BATCH_SIZE

MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "cnn_model.h5"
CLASSES_PATH = MODELS_DIR / "classes.json"

def create_gens():
    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    val_gen = val_aug.flow_from_directory(
        DATA_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return train_gen, val_gen

def build_model(num_classes):
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base

if __name__ == "__main__":
    train_gen, val_gen = create_gens()
    num_classes = train_gen.num_classes
    classes = {v:k for k,v in train_gen.class_indices.items()}
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)
    print("Classes:", classes)

    model, base = build_model(num_classes)
    checkpoint = ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy", save_best_only=True, verbose=1)
    early = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)

    model.fit(
        train_gen,
        steps_per_epoch=max(1, train_gen.samples // BATCH_SIZE),
        validation_data=val_gen,
        validation_steps=max(1, val_gen.samples // BATCH_SIZE),
        epochs=12,
        callbacks=[checkpoint, early]
    )
    print(f"Saved CNN model to {MODEL_PATH}")

    # Optional fine-tune
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=4,
        callbacks=[checkpoint, early]
    )
    print("Fine-tuning done.")
