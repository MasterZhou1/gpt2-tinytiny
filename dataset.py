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
    save_file_path = os.path.join(save_path, f'{save_name}.pt')
    torch.save(final_tensor, save_file_path)
    t1 = time.time()
    print(f'Process finished with {t1-t0:.3f} seconds.')


def preprocess_and_save_sft(dataset, tokenizer, block_size, save_name, save_path='./data'):
    # Create the save folder if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    t0 = time.time()

    # Lists to store preprocessed Q and A tensors
    q_tensors = []
    a_tensors = []

    for i, text in enumerate(dataset):
        q_tokens = []
        a_tokens = []
        for example in text['messages']:
            role = example['role']
            content = example['content']

            # Tokenize the content
            input_ids = tokenizer.encode(content, return_tensors="pt").flatten().to(torch.int32)

            # Check if the length exceeds block_size
            if len(input_ids) > block_size:
                # Skip this example
                break

            # Decide whether the example is a question (Q) or an answer (A)
            if role == "user":
                q_tokens.append(input_ids)
            elif role == "assistant":
                a_tokens.append(input_ids)

        # Append tokens to the main lists
        if q_tokens and a_tokens:
            q_tensors.extend(q_tokens)
            a_tensors.extend(a_tokens)

        if i % 5000 == 0:
            print(f'[{i}/{len(dataset)}] processed.')
            # Print the number of examples for questions and answers
            # print(f'Number of examples for questions: {len(q_tensors)}')
            # print(f'Number of examples for answers: {len(a_tensors)}')

    assert len(q_tensors) == len(a_tensors), \
        f'Q {len(q_tensors)} and A {len(a_tensors)} size didn\'t match '

    # Save the final tensors to files
    q_save_file_path = os.path.join(save_path, f'{save_name}_Q_sft.pt')
    a_save_file_path = os.path.join(save_path, f'{save_name}_A_sft.pt')

    torch.save(q_tensors, q_save_file_path)
    torch.save(a_tensors, a_save_file_path)

    t1 = time.time()
    print(f'Process finished with {t1 - t0:.3f} seconds.')

if __name__ == '__main__':
    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # load pretrain dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    # Access the splits
    train_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']

    preprocess_and_save(validation_data, tokenizer, save_name='validation_data')
    preprocess_and_save(test_data, tokenizer, save_name='test_data')
    preprocess_and_save(train_data, tokenizer, save_name='train_data')

    # load sft dataset
    dataset = load_dataset("arazd/tulu_cot")

    # Access the splits
    sft_data = dataset['train']

    preprocess_and_save_sft(sft_data, tokenizer, block_size=256, save_name='train')
