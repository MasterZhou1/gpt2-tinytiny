# gpt2-tinytiny
## [Final Report](gpt2_tinytiny.pdf)

Learning the GPT2 architecture and pipeline of LLMs from scratch.

This is a course final project for ESE5460 Principals of Deep Learning in University of Pennsylvania.

We have implemented and finished the pretrian and supervised-finetuning (SFT) part of GPT2-tinytiny.

### SFT format:

```
<|user|>
Your message here!
<|assistant|>
```

Include a newline '\n' after <|assistant|>, this could affect generation quality.

### Usage:
- This project mainly uses huggingface, loralib and pytorch. You can find the versions in `requirements.txt`.
- Need to clearly specify hyperparams `(lr, iters, model_type...)` in `train.py` and `train_sft.py`.
- `prompts.txt` is the test set questions from LIMA. Besides, LIMA is a gated dataset in huggingface, 
  in order to use it you need to download it from the website (link can found below) and 
  put it to the correct folder (you might need to look at the `dataset.py` for clearer infomation).

### Findings:
- LORA appears to exhibit inferior performance compared to retrain_all, possibly owing to the model's smaller size. 
  The current model is intentionally kept small for learning purposes and due to limited computational resources. It's approximately half the size of GPT-2 small.
- Despite the smaller model, its performance is somewhat amazing. It does manage to generate semantically correct sentences and retains a rudimentary memory of knowledge within the dataset.
- To enhance the model's capabilities, exploring larger hyperparameters and incorporating a more extensive dataset, such as PILE, could be considered.
- Diversity of dataset can affect model's overfitting behavior during SFT. Using Tulu SFT dataset has no overfitting but LIMA has.
### Main References: 

model: https://github.com/karpathy/nanoGPT/tree/master

Wikitext103 dataset: https://huggingface.co/datasets/wikitext

Tulu dataset: https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture

LIMA dataset: https://huggingface.co/datasets/GAIR/lima

how chatgpt works: https://www.assemblyai.com/blog/how-chatgpt-actually-works/

SFT: https://cameronrwolfe.substack.com/p/understanding-and-using-supervised

LLMs course: https://stanford-cs324.github.io/winter2022/lectures/introduction/

