import json

def update_category_id(json_path, output_path):
    """
    Updates 'category_id' from 0 to 1 in the given JSON file.

    Args:
        json_path (str): Path to the input JSON file.
        output_path (str): Path to save the updated JSON file.
    """
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Update category_id in annotations
    for annotation in data:
        if annotation["category_id"] == 0:
            annotation["category_id"] = 1

    # Save updated JSON file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Updated JSON file saved to {output_path}")

# Example usage
input_json = "epoch23_predictions.json"
output_json = "epoch23_predictions_updated.json"
update_category_id(input_json, output_json)