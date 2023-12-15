from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
import torch
import time

def preprocess_and_save(dataset, tokenizer, save_name, save_path='./data'):
    # Create the save folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    t0 = time.time()
    # List to store preprocessed text tensors
    processed_tensors = []

    for i, example in enumerate(dataset):
        text = example['text']

        # Encode the text
        # Vocabulary Size: 50257 << uint16=65535. However torch doesn't support unit16, so use int32 instead
        input_ids = tokenizer.encode(text, return_tensors="pt").flatten().to(torch.int32)

        # Append the tensor to the list
        processed_tensors.append(input_ids)

        if i % 2000 == 0:
            print(f'[{i}/{len(dataset)}] processed.')

    # Concatenate all tensors into a single tensor
    final_tensor = torch.cat(processed_tensors, dim=0)

    # Save the final tensor to a file
    save_file_path = os.path.join(save_path, save_name)
    torch.save(final_tensor, save_file_path)
    t1 = time.time()
    print(f'Process finished with {t1-t0:.3f} seconds.')


if __name__ == '__main__':
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Access the splits
    train_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']

    preprocess_and_save(validation_data, tokenizer, save_name='validation_data.pt')
    preprocess_and_save(test_data, tokenizer, save_name='test_data.pt')
    preprocess_and_save(train_data, tokenizer, save_name='train_data.pt')

