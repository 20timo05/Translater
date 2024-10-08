import pickle
import random
import os
from tqdm import tqdm
import torch
import multiprocessing as mp

from Tokenizer import RegexTokenizer
from parameters import GPT4_SPLIT_PATTERN, NUM_MERGES, SPECIAL_TOKENS, CONTEXT_SIZE

""" Train/ Load Tokenizer & load/ preprocess the whole dataset"""
def tokenize_sentence_pair(args):
    pair, tokenizer, pad_token, start_token, end_token = args
    english, german = pair
    eng_enc, ger_enc = tokenizer.encode(english), tokenizer.encode(german)

    # Truncation and Padding for english sentences (ensure len == CONTEXT_SIZE)
    if len(ger_enc) + 2 > CONTEXT_SIZE:
        ger_enc = ger_enc[-(CONTEXT_SIZE - 2) :]

    ger_enc = [start_token] + ger_enc + [end_token]
    
    if len(ger_enc) <= CONTEXT_SIZE:
        ger_enc = ger_enc + [pad_token] * (CONTEXT_SIZE - len(ger_enc) + 1)

    if len(eng_enc) <= CONTEXT_SIZE:
        eng_enc = eng_enc + [pad_token] * (CONTEXT_SIZE - len(eng_enc))
    else:
        eng_enc = eng_enc[-CONTEXT_SIZE:]

    return eng_enc, ger_enc


def tokenize_dataset(translations, tokenizer):
    pad_token = [key for key, value in tokenizer.vocab.items() if value == b"<|PAD|>"][0]
    start_token = [
        key for key, value in tokenizer.vocab.items() if value == b"<|STARTOFTEXT|>"
    ][0]
    end_token = [
        key for key, value in tokenizer.vocab.items() if value == b"<|ENDOFTEXT|>"
    ][0]

    args = [(pair, tokenizer, pad_token, start_token, end_token) for pair in translations]

    with mp.Pool(mp.cpu_count()) as pool:
        results = list(
            tqdm(pool.imap(tokenize_sentence_pair, args), total=len(translations))
        )

    eng_data, ger_data = zip(*results)
    return list(eng_data), list(ger_data)


""" Train Tokenizer, then preprocess & tokenize dataset"""
def train(translations):
    # Tokenization
    tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
    tokenizer_file_path = "models/tokenizer.pkl"

    print("Train Tokenizer...")
    # only use a subset of the original dataset for tokenizer training
    translations_subset = random.sample(translations, int(len(translations) * 0.01))
    tokenizer.train(translations_subset, num_merges=NUM_MERGES)

    for st in SPECIAL_TOKENS:
        tokenizer.vocab[max(tokenizer.vocab) + 1] = st

    print("Preprocess & Tokenize dataset...")
    eng_data, ger_data = tokenize_dataset(translations, tokenizer)

    # Save the combined dictionary to a JSON file
    with open(tokenizer_file_path, "wb") as f:
        pickle.dump(
            {
                "vocab": tokenizer.vocab,
                "merges": tokenizer.merges,
                "eng_data": eng_data,
                "ger_data": ger_data,
            },
            f,
        )


def load_tokenizer_and_dataset(filepath):
    assert os.path.exists(filepath), "Cannot find Tokenizer"

    tokenizer = RegexTokenizer(GPT4_SPLIT_PATTERN)
    with open(filepath, "rb") as f:
        combined_dict = pickle.load(f)

    # Extract vocab and merges dictionaries from the combined dictionary
    tokenizer.vocab = combined_dict["vocab"]
    tokenizer.merges = combined_dict["merges"]

    # Load preprocessed & tokenized dataset
    eng_data = combined_dict["eng_data"]
    ger_data = combined_dict["ger_data"]

    eng_data, ger_data = torch.tensor(eng_data), torch.tensor(ger_data)
    return (tokenizer, eng_data, ger_data)


if __name__ == "__main__":
    with open("data/translation.txt", "r", encoding="utf-8") as f:
        data = f.read().splitlines()

    # load data
    translations = []
    for sample in data:
        english, german, src = sample.split("\t")
        translations.append((english, german))

    train(translations)
