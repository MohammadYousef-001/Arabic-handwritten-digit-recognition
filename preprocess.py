import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ANGKA ARAB")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data")

TARGET_SIZE = 64
DIGIT_SIZE = 52
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

MARGIN = 4


def is_image_file(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)


def get_class_folders(dataset_path):
    folders = []
    for item in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, item)
        if os.path.isdir(full_path):
            folders.append(item)
    return sorted(folders, key=lambda x: int(x) if x.isdigit() else x)


def rgba_to_white_background(img):
    if img.mode == "RGBA":
        white_bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white_bg, img)
        img = img.convert("RGB")
    else:
        img = img.convert("RGB")
    return img


def crop_digit_otsu(gray_img, margin=4):
    """
    Crop digit region using Otsu thresholding.
    gray_img must be a PIL grayscale image.
    """
    arr = np.array(gray_img)

    # Otsu automatically finds the threshold for this image
    # THRESH_BINARY_INV makes the digit white in the binary mask
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.argwhere(binary > 0)

    # if nothing is found, return original
    if coords.size == 0:
        return gray_img

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)
    y_max = min(arr.shape[0] - 1, y_max + margin)
    x_max = min(arr.shape[1] - 1, x_max + margin)

    cropped = gray_img.crop((x_min, y_min, x_max + 1, y_max + 1))
    return cropped


def resize_and_center(img, digit_size=52, target_size=64):
    """
    Resize while preserving aspect ratio to fit inside digit_size x digit_size,
    then center inside target_size x target_size white canvas.
    """
    w, h = img.size

    scale = min(digit_size / w, digit_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (target_size, target_size), 255)

    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2

    canvas.paste(img, (paste_x, paste_y))
    return canvas


def preprocess_image(file_path):
    with Image.open(file_path) as img:
        original = img.copy()

        # Step 1: remove transparency safely
        img = rgba_to_white_background(img)

        # Step 2: grayscale
        gray = img.convert("L")

        # Step 3: crop digit region using Otsu
        cropped = crop_digit_otsu(gray, margin=MARGIN)

        # Step 4: resize and center inside 64x64
        final_img = resize_and_center(cropped, digit_size=DIGIT_SIZE, target_size=TARGET_SIZE)

        # Step 5: normalize
        final_array = np.array(final_img, dtype=np.float32) / 255.0

    return original, gray, cropped, final_img, final_array


def load_dataset(dataset_path):
    X = []
    y = []
    preview_items = []

    class_folders = get_class_folders(dataset_path)

    for class_name in class_folders:
        class_path = os.path.join(dataset_path, class_name)
        first_saved = False

        for file_name in os.listdir(class_path):
            if not is_image_file(file_name):
                continue

            file_path = os.path.join(class_path, file_name)

            try:
                original, gray, cropped, final_img, final_array = preprocess_image(file_path)

                X.append(final_array)
                y.append(int(class_name))

                if not first_saved:
                    preview_items.append({
                        "label": int(class_name),
                        "original": original,
                        "gray": gray,
                        "cropped": cropped,
                        "final_img": final_img,
                        "final_array": final_array
                    })
                    first_saved = True

            except Exception as e:
                print(f"Error with file: {file_path}")
                print(e)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, preview_items


def save_preview(preview_items, save_path):
    rows = len(preview_items)
    cols = 4

    plt.figure(figsize=(12, 3 * rows))

    for i, item in enumerate(preview_items):
        plt.subplot(rows, cols, i * cols + 1)
        plt.imshow(item["original"])
        plt.title(f"Class {item['label']} - Original")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 2)
        plt.imshow(item["gray"], cmap="gray")
        plt.title("Grayscale")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 3)
        plt.imshow(item["cropped"], cmap="gray")
        plt.title("Cropped (Otsu)")
        plt.axis("off")

        plt.subplot(rows, cols, i * cols + 4)
        plt.imshow(item["final_img"], cmap="gray")
        plt.title("Final 64x64")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y, preview_items = load_dataset(DATASET_PATH)

    print("=" * 50)
    print("PREPROCESSING SUMMARY")
    print("=" * 50)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Pixel min: {X.min():.4f}")
    print(f"Pixel max: {X.max():.4f}")

    # Save arrays
    x_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")
    np.save(x_path, X)
    np.save(y_path, y)

    # Save preview figure
    preview_path = os.path.join(OUTPUT_DIR, "preprocessing_preview.png")
    save_preview(preview_items, preview_path)

    print("\nSaved files:")
    print(f"X saved to: {x_path}")
    print(f"y saved to: {y_path}")
    print(f"Preview saved to: {preview_path}")


if __name__ == "__main__":
    main()