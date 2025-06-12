import os
from collections import defaultdict
import matplotlib.pyplot as plt

def count_images_by_set(image_folder):
    """
    Counts the number of images in each set within the given folder.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        dict: A dictionary with set names as keys and image counts as values.
    """
    # Dictionary to store counts
    set_counts = defaultdict(int)

    # Iterate through files in the folder
    for filename in os.listdir(image_folder):
        if filename.startswith("set") and filename.endswith((".jpg", ".png", ".jpeg")):  # Adjust extensions as needed
            set_name = filename.split("_")[0]  # Extract set name (e.g., "set01")
            set_counts[set_name] += 1

    return dict(set_counts)

def plot_set_counts(set_counts):
    """
    Plots the number of images per set as a bar chart.

    Args:
        set_counts (dict): Dictionary with set names as keys and image counts as values.
    """
    # Sort sets by name
    sorted_sets = sorted(set_counts.items())
    set_names, counts = zip(*sorted_sets)

    # Create bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(set_names, counts, color='skyblue')
    plt.xlabel('Set Name')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Set')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Example usage
image_folder = "/home/chanwoo/git/deeplearning_2025_1/AUE8088/datasets/kaist-rgbt/test/images/visible"
set_counts = count_images_by_set(image_folder)

# Print results
for set_name, count in set_counts.items():
    print(f"{set_name}: {count} images")

# Plot results
plot_set_counts(set_counts)