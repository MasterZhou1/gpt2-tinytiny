from datasets import load_dataset
from transformers import GPT2Tokenizer
import os
import torch
import time


def preprocess_and_save(dataset, tokenizer, save_name, prompt_formatting, dataset_type, save_path='./data'):
    # Create the save folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    t0 = time.time()
    # List to store preprocessed text tensors
    processed_tensors = []

    for i, example in enumerate(dataset):
        text = prompt_formatting(example, dataset_type)

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
    save_file_path = os.path.join(save_path, f'{save_name}.pt')
    torch.save(final_tensor, save_file_path)
    t1 = time.time()
    print(f'Process finished with {t1-t0:.3f} seconds.')


def prompt_formatting(example, dataset_type=None):
    if dataset_type == 'wikitext103':
        # wikitext103 data: {..., 'text':...}
        return example['text']
    elif dataset_type == 'tulu':
        # tulu data: {...,'message':[{ "role": "user", "content":...}, { "role": "assistant", "content":...}]}
        text = ''
        for t in example['messages']:
            if t['role'] == 'user':
                text += f"<|user|>\n{t['content']}\n"
            elif t['role'] == 'assistant':
                text += f"<|assistant|>\n{t['content']}\n"
        return text
    elif dataset_type == 'lima':
        # tulu data: {...,'conversations':["...", "..."]}
        text = ''
        for i, t in enumerate(example['conversations']):
            if i % 2 == 0:
                text += f"<|user|>\n{t}\n"
            else:
                text += f"<|assistant|>\n{t}\n"
        return text

if __name__ == '__main__':
    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load pretrain dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Access the splits
    validation_data = dataset['validation']
    test_data = dataset['test']
    train_data = dataset['train']

    preprocess_and_save(validation_data, tokenizer, 'validation_data', prompt_formatting, 'wikitext103')
    preprocess_and_save(test_data, tokenizer, 'test_data', prompt_formatting, 'wikitext103')
    preprocess_and_save(train_data, tokenizer, 'train_data', prompt_formatting, 'wikitext103')

    # load sft dataset
    # dataset = load_dataset("arazd/tulu_cot") # remember to change dataset_type == 'tulu
    dataset = load_dataset("./data/lima", cache_dir='./data/lima')

    validation_percentage = 0.05  # Adjust this based on your needs
    # Access the splits
    sft_data = dataset['train'].train_test_split(test_size=validation_percentage)

    sft_train = sft_data['train']
    sft_validation = sft_data['test']
    preprocess_and_save(sft_train, tokenizer, 'train_sft', prompt_formatting, 'lima')
    preprocess_and_save(sft_validation, tokenizer, 'validation_sft', prompt_formatting, 'lima')


