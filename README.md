# gpt2-tinytiny
## [Final Report](gpt2_tinytiny.pdf)

Learning the gpt2 structure from scratch.

This is a course final project for ESE5460 Principals of Deep Learning in University of Pennsylvania.

We implement and finish the pretrian and supervised-finetuning (SFT) part of GPT2-tinytiny.

### SFT format:

```
<|user|>
Your message here!
<|assistant|>
```

Include a newline '\n' after <|assistant|>, this will affect generation quality.


### Main References: 

model: https://github.com/karpathy/nanoGPT/tree/master

Wikitext103 data: https://huggingface.co/datasets/wikitext

Tulu data: https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture

how chatgpt works: https://www.assemblyai.com/blog/how-chatgpt-actually-works/

SFT: https://cameronrwolfe.substack.com/p/understanding-and-using-supervised

LLM course: https://stanford-cs324.github.io/winter2022/lectures/introduction/

