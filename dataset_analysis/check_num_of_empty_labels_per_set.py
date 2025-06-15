import os
from collections import defaultdict
import matplotlib.pyplot as plt

def count_labels_by_set(label_folder):
    """
    Counts the number of empty and filled label files in each set within the given folder.

    Args:
        label_folder (str): Path to the folder containing label files.

    Returns:
        dict: A dictionary with set names as keys and a tuple (filled_count, empty_count) as values.
    """
    # Dictionary to store counts
    set_counts = defaultdict(lambda: [0, 0])  # [filled_count, empty_count]

    # Iterate through files in the folder
    for filename in os.listdir(label_folder):
        if filename.startswith("set") and filename.endswith(".txt"):  # Only process .txt files
            set_name = filename.split("_")[0]  # Extract set name (e.g., "set01")
            file_path = os.path.join(label_folder, filename)

            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                set_counts[set_name][1] += 1  # Increment empty count
            else:
                set_counts[set_name][0] += 1  # Increment filled count

    return dict(set_counts)

def plot_label_counts(set_counts):
    """
    Plots the number of empty and filled label files per set as a bar chart.

    Args:
        set_counts (dict): Dictionary with set names as keys and (filled_count, empty_count) as values.
    """
    # Sort sets by name
    sorted_sets = sorted(set_counts.items())
    set_names, counts = zip(*sorted_sets)
    filled_counts = [count[0] for count in counts]
    empty_counts = [count[1] for count in counts]

    # Create bar chart
    x = range(len(set_names))
    bar_width = 0.4
    plt.figure(figsize=(12, 6))
    plt.bar([i - bar_width / 2 for i in x], filled_counts, width=bar_width, label='Filled Labels', color='skyblue')
    plt.bar([i + bar_width / 2 for i in x], empty_counts, width=bar_width, label='Empty Labels', color='salmon')
    plt.xticks(x, set_names, rotation=45)
    plt.xlabel('Set Name')
    plt.ylabel('Number of Label Files')
    plt.title('Number of Empty and Filled Label Files per Set')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
label_folder = "datasets/kaist-rgbt/train/labels"
set_counts = count_labels_by_set(label_folder)

# Print results
for set_name, (filled, empty) in set_counts.items():
    print(f"{set_name}: Filled Labels = {filled}, Empty Labels = {empty}")

# Plot results
plot_label_counts(set_counts)