import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
PATIENCE = 5
RANDOM_STATE = 42


def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # add channel dimension for CNN input
    X = X[..., np.newaxis]


    return X, y


def split_data(X, y):
    # the spllit is 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    
    print("DATA SPLIT:")
    
    print(f"X_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_val shape   : {y_val.shape}")
    print(f"y_test shape  : {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model():
    model = Sequential([
        Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)),

        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            kernel_initializer="he_normal"
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(
            128,
            activation="relu",
            kernel_initializer="he_normal"
        ),
        BatchNormalization(),
        Dropout(0.5),

        Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")


    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path.replace("accuracy", "loss"), dpi=200)
    plt.close()


def plot_confusion_matrix_figure(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")


    plt.tight_layout()

    plt.savefig(save_path, dpi=200)
    plt.close()


def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, digits=4)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    model = build_model()

    
    print("MODEL SUMMERY:")
    
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    
    print("TEST RESULTS:")
    
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save plots and reports
    accuracy_plot_path = os.path.join(MODEL_DIR, "training_accuracy.png")

    confusion_matrix_path = os.path.join(MODEL_DIR, "confusion_matrix.png")

    report_path = os.path.join(MODEL_DIR, "classification_report.txt")

    final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
    plot_training_history(history, accuracy_plot_path)
    plot_confusion_matrix_figure(y_test, y_pred, confusion_matrix_path)
    save_classification_report(y_test, y_pred, report_path)
    model.save(final_model_path)

    print("\nsaved files:")
    print(os.path.join(MODEL_DIR, "best_model.keras"))
    print(final_model_path)
    print(accuracy_plot_path)
    print(os.path.join(MODEL_DIR, "training_loss.png"))
    print(confusion_matrix_path)
    print(report_path)


if __name__ == "__main__":
    main()