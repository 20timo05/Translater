from tqdm import tqdm
import regex as re

class RegexTokenizer:
    def __init__(self, regex_split_pattern):
        self.regex = regex_split_pattern

        self.vocab = {}
        self.merges = {}

    def train(self, text_samples, vocab_size, verbose=False):
        # adjust tokenizer to not accept a string of text, but a list of strings
        split_text = [
            [list(map(int, chunk.encode("utf-8")))
            for chunk in re.findall(self.regex, sentence)]
            for sample in text_samples
            for sentence in sample
        ]
        split_tokens = [token for sentence_tokens in split_text for token in sentence_tokens]
        print("Preprocessing Done", len(split_tokens))

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size-256

        for i in (tqdm(range(num_merges)) if not verbose else range(num_merges)):
            # get count of each possible bigram in the train data
            stats = self._get_stats(split_tokens)
            # get bigram that occured most often
            max_pair, new_token = max(stats, key=stats.get), 256+i
            split_tokens = self._merge_pairs(split_tokens, max_pair, new_token)
            
            merges[max_pair] = new_token
            vocab[new_token] = vocab[max_pair[0]] + vocab[max_pair[1]]
            
            if verbose:
                print(f"merge {i+1}/{num_merges}: {max_pair} -> {new_token} ({vocab[new_token]}) had {stats[max_pair]} occurrences")
        
        self.vocab = vocab
        self.merges = merges

    def encode(self, text):
        tokens = list(map(int, text.encode("utf-8")))

        while len(tokens) >= 2:
            counts = self._get_stats(tokens)
            # find best bigram to merge (first in merges list)
            pair = min(counts, key=lambda bigram: self.merges.get(bigram, float("inf")))       
            
            # no bigrams left to merge
            if pair not in self.merges:
                break
            
            new_token = self.merges[pair]
            tokens = self._merge_pairs(tokens, pair, new_token)
        
        return tokens

    def decode(self, tokens):
        tokens = b"".join([self.vocab[i] for i in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, tokens):
        counts = {}

        # check if tokens is 2d list (split tokens) or 1d
        if isinstance(tokens[0], list):
            bigrams = [bigram for split in tokens for bigram in zip(split, split[1:])]
        else:
            bigrams = zip(tokens, tokens[1:])
    
        for bigram in bigrams:
            counts[bigram] = counts.get(bigram, 0) + 1
        return counts
    
    def _merge_pairs(self, tokens, pair, new_token):
        def merge(tokens):
            merged_tok = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    merged_tok.append(new_token)
                    i += 1
                else:
                    merged_tok.append(tokens[i])
                i += 1
            
            return merged_tok

        # check if tokens is 2d list (split tokens) or 1d
        if isinstance(tokens[0], list):
            split_merged_tokens = [merge(split) for split in tokens]
            return split_merged_tokens
        
        else:
            return merge(tokens)