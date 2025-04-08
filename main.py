import os
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.image_recognition import SetClipModel
from src.poem_generator import poem_generator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "poet-gpt2")


def load_model(model_path, model_name="gpt2"):
    """
    Load a pretrained GPT2 model and tokenizer locally or download if not present.
    Args:
        model_path (str): Path to the directory where the model is (or will be) stored.
        model_name (str): Name of the model to download if not present locally.
    Returns:
        model (GPT2LMHeadModel): The language model.
        tokenizer (GPT2Tokenizer): The corresponding tokenizer.
    """

    model_file = os.path.join(model_path, "model.safetensors")

    if not os.path.exists(model_file):
        print("Model not found, downloading...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        os.makedirs(model_path, exist_ok=True)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print("Mode saved!")
    else:
        print("Model found, loading...")

    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

  
model, tokenizer = load_model(MODEL_PATH)


def load_labels(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['labels']
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found : {file_path}")
    except json.JSONDecodeError:
        raise ValueError("JSON file decoding error.")


labels = load_labels(os.path.join(DATA_DIR, "labels.json"))


def generate_poem_from_picture(image_path, labels):
    """
    Args:
        image_path (str): path to picture.
        labels (list of str): List of possible thematic labels to be detected in the image.
    Returns:
        str: generate a poem according to the visual content of the picture.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The specified image does not exist : {image_path}")

    theme = SetClipModel().image_label_detector(image_path, labels)
    poem = poem_generator(
        model_path=MODEL_PATH,
        theme=theme,
        max_length=200,
        temperature=0.5,
        top_k=60,
        top_p=0.9,
        repetition_penalty=1.5
    )
    return theme, poem


def main():
    image_path = os.path.join(DATA_DIR, "images", "sunset.jpg")
    try:
        print("\nGeneration in progress...")
        theme, poem = generate_poem_from_picture(image_path, labels)
        print("\n", theme)
        print("\n", poem)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
