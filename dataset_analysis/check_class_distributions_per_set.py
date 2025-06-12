import os
from collections import defaultdict
import matplotlib.pyplot as plt

# Define class names
CLASS_NAMES = {
    0: "person",
    1: "cyclist",
    2: "people",
    3: "person?"
}

def count_labels_by_class(label_folder):
    """
    Counts the number of labels per class (0, 1, 2, 3) for each set.

    Args:
        label_folder (str): Path to the folder containing label files.

    Returns:
        dict: A dictionary with set names as keys and a dictionary of class counts as values.
    """
    # Dictionary to store counts
    set_counts = defaultdict(lambda: defaultdict(int))  # {set_name: {class_id: count}}

    # Iterate through files in the folder
    for filename in os.listdir(label_folder):
        if filename.startswith("set") and filename.endswith(".txt"):  # Only process .txt files
            set_name = filename.split("_")[0]  # Extract set name (e.g., "set01")
            file_path = os.path.join(label_folder, filename)

            # Process non-empty files
            if os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    for line in f:
                        class_id = int(line.split()[0])  # Extract the first value (class ID)
                        if class_id in CLASS_NAMES:  # Ensure class_id is valid
                            set_counts[set_name][class_id] += 1

    return dict(set_counts)

def plot_class_counts(set_counts):
    """
    Plots the number of labels per class for each set as a bar chart.

    Args:
        set_counts (dict): Dictionary with set names as keys and class counts as values.
    """
    # Sort sets by name
    sorted_sets = sorted(set_counts.items())
    set_names = [set_name for set_name, _ in sorted_sets]
    counts = [count_dict for _, count_dict in sorted_sets]

    # Prepare data for plotting
    class_counts = {class_name: [] for class_name in CLASS_NAMES.values()}
    for count_dict in counts:
        for class_id, class_name in CLASS_NAMES.items():
            class_counts[class_name].append(count_dict.get(class_id, 0))

    # Create bar chart
    x = range(len(set_names))
    bar_width = 0.2
    plt.figure(figsize=(12, 6))

    for i, (class_name, values) in enumerate(class_counts.items()):
        plt.bar([j + i * bar_width for j in x], values, width=bar_width, label=class_name)

    plt.xticks([j + (len(class_counts) - 1) * bar_width / 2 for j in x], set_names, rotation=45)
    plt.xlabel('Set Name')
    plt.ylabel('Number of Labels')
    plt.title('Number of Labels per Class for Each Set')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
label_folder = "/home/chanwoo/git/deeplearning_2025_1/AUE8088/datasets/kaist-rgbt/train/labels"
set_counts = count_labels_by_class(label_folder)

# Print results
for set_name, class_count in set_counts.items():
    print(f"{set_name}: {dict(class_count)}")

# Plot results
plot_class_counts(set_counts)