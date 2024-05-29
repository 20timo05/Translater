from tqdm import tqdm
import regex as re

class RegexTokenizer:
    def __init__(self, regex_split_pattern):
        self.regex = regex_split_pattern

        self.vocab = {}
        self.merges = {}

    def train(self, text, vocab_size, verbose=False):
        # convert text to utf-8 bytes and then each byte to integers. ex.: "ab" => [97, 98]
        split_text = re.findall(self.regex, text)
        split_tokens = [list(map(int, t.encode("utf-8"))) for t in split_text]

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
        split_text = re.findall(self.regex, text)
        split_tokens = [list(map(int, t.encode("utf-8"))) for t in split_text]

        for pair, new_token in self.merges.items():
            split_tokens = self._merge_pairs(split_tokens, pair, new_token)
        
        # concat split token sequence back together
        return [t for tokens in split_tokens for t in tokens]
    
    def decode(self, tokens):
        tokens = b"".join([self.vocab[i] for i in tokens])
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, split_tokens):
        counts = {}

        bigrams = [bigram for tokens in split_tokens for bigram in zip(tokens, tokens[1:])]
        for bigram in bigrams:
            counts[bigram] = counts.get(bigram, 0) + 1
        return counts
    
    def _merge_pairs(self, split_tokens, pair, new_token):
        split_merged_tokens = []

        for tokens in split_tokens:
            
            merged_tok = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    merged_tok.append(new_token)
                    i += 1
                else:
                    merged_tok.append(tokens[i])
                i += 1

            split_merged_tokens.append(merged_tok)

        return split_merged_tokens