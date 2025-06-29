import json
import os

def load_json(filename: str) -> dict:
    """
    Loads a JSON file and returns its content as a dictionary.

    Parameters:
        filename (str): The path to the JSON file.

    Returns:
        dict: The content of the JSON file.
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_json(data: dict, filename: str) -> None:
    """
    Saves a dictionary to a JSON file.

    Parameters:
        data (dict): The data to save.
        filename (str): The path to the file where the data will be saved.
    """
    if not filename.endswith('.json'):
        raise ValueError("Filename must end with .json")
    
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)