import os
from collections import Counter
from statistics import mean
from PIL import Image
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ANGKA ARAB")

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(filename):
    return filename.lower().endswith(VALID_EXTENSIONS)


def get_class_folders(dataset_path):
    folders = []
    for item in os.listdir(dataset_path):
        full_path = os.path.join(dataset_path, item)
        if os.path.isdir(full_path):
            folders.append(item)
    return sorted(folders, key=lambda x: int(x) if x.isdigit() else x)

# get summary and info about the dataset and plot number of images per class
def main():
    class_folders = get_class_folders(DATASET_PATH)

    total_images = 0
    non_grayscale_count = 0
    unreadable_count = 0

    widths = []
    heights = []
    width_counter = Counter()
    height_counter = Counter()
    mode_counter = Counter()
    class_counts = {}

    for class_name in class_folders:
        class_path = os.path.join(DATASET_PATH, class_name)
        class_image_count = 0

        for file_name in os.listdir(class_path):
            if not is_image_file(file_name):
                continue

            file_path = os.path.join(class_path, file_name)

            try:
                with Image.open(file_path) as img:
                    total_images += 1
                    class_image_count += 1

                    width, height = img.size
                    widths.append(width)
                    heights.append(height)

                    width_counter[width] += 1
                    height_counter[height] += 1
                    mode_counter[img.mode] += 1

                    if img.mode != "L":
                        non_grayscale_count += 1

            except Exception:
                unreadable_count += 1

        class_counts[class_name] = class_image_count

    if total_images == 0:
        print("No images found.")
        return

    avg_width = mean(widths)
    avg_height = mean(heights)

    most_common_width, most_common_width_count = width_counter.most_common(1)[0]
    most_common_height, most_common_height_count = height_counter.most_common(1)[0]

    print("=" * 50)
    print("DATA SUMMARY")
    print("=" * 50)
    print(f"Total readable images       : {total_images}")
    print(f"Unreadable images           : {unreadable_count}")
    print(f"Non-grayscale images        : {non_grayscale_count}")
    print()
    print(f"Average width               : {avg_width:.2f}")
    print(f"Average height              : {avg_height:.2f}")
    print()
    print(f"Most common width (mode)    : {most_common_width}  -> {most_common_width_count} images")
    print(f"Most common height (mode)   : {most_common_height} -> {most_common_height_count} images")
    print()
    print("Image mode counts:")
    for img_mode, count in mode_counter.items():
        print(f"  {img_mode}: {count}")

    
    # plot number of images per class
  
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_counts.keys(), class_counts.values())

    plt.title("Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.ylim(0, max(class_counts.values()) + 80)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()