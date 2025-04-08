import numpy as np
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class IntializePoemModel:
    def __init__(self, model_path):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")


def create_prompt(theme):
    start_of_promt = [
        "For I am the",
        "I only I could have the",
        "Then we see the"
    ]
    # Set up the initial prompt
    num_start_of_prompt = np.random.randint(0, 3)
    prompt = f"{start_of_promt[num_start_of_prompt]} {theme},"

    return prompt


def poem_generator(
    model_path="trained_model/poet-gpt2",
    theme="moon",
    max_length=200,
    temperature=0.5,
    top_k=60,
    top_p=0.9,
    repetition_penalty=1.5
):
    """
    Take in input the poet_gpt2 model path, its parameters and a theme and generate a poem.
    """
    poet_gpt2 = IntializePoemModel(model_path)
    prompt = create_prompt(theme)

    input_ids = poet_gpt2.tokenizer.encode(prompt, return_tensors="pt")

    # Generate text
    output = poet_gpt2.model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
    )

    # Decode and print the poem
    poem = poet_gpt2.tokenizer.decode(output[0], skip_special_tokens=True)

    return poem
