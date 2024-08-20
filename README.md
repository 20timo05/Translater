# Translating English to German
The goal of the GPT is to learn how to translate text from English to German. The architecture is based on the famous Paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762). In theory, the GPT should learn the underlying sentence structures and meaning of tokens through self-attention.

# Purpose
This GPT is obviously not intended for any Real-World use, but it is a great learning Project to put everything I have learned about NLP from Andrej Karpathy's Video Lectures, specifically the [Neural Networks: Zero to Hero](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=A7wmUXohjjnHA6My) playlist and 3Blue1Brown's [Video on Self-Attention](https://youtu.be/eMlx5fFNoYc?si=xZJUevr2iAHhz_yM).

# Starter Code
The starter code is my own implementation of the Transformer Architecture similar to [this one](https://youtu.be/kCc8FmEb1nY). It is trained on a concatenation of all Harry Potter Books and therefore is only able to generate Harry Potter like Text (see: `./output/HarryPotterText.txt`) (although the text does not make sense).


# Next Steps

Since the Transformer model hasn’t yet met my performance goals, I’ll focus on refining both the encoder and decoder. I plan to pretrain the encoder with "Masked Language Modeling" to improve its context understanding, and the decoder with "(autoregressive) language modeling" for better sequence generation. After these pretraining steps, I will reevaluate and fine-tune the model.

Additional optimizations include:
- ✔ Weight Tying: Input Embeddings and lm_head => save parameters => faster training
- Normalization for residual pathways at initialization
- Mixed Precision (for matmul & )
- Flash Attention
- "Nice" Numbers
- Gradient clipping
- weight decay
- FusedAdamW
- Big Batch Size with Gradient Accumulation
- Distributed Data Parallel (use multiple GPUs like multiprocessing)

I will also transition to cloud-based training with multiple GPUs to leverage greater computational power.