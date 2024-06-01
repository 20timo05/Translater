import torch

# Hyperparameters
NUM_MERGES = 10000
SPECIAL_TOKENS = [b"<|ENDOFTEXT|>", b"<|PAD|>"]
VOCAB_SIZE = 256 + NUM_MERGES + len(SPECIAL_TOKENS)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

EMBEDDING_SIZE = 32
CONTEXT_SIZE = 16
BATCH_SIZE = 32
MAX_STEPS = 5000
LEARNING_RATE = 3E-4
ENCODER_BLOCK_COUNT = 2
DECODER_BLOCK_COUNT = 2
NUM_HEADS = 2
DROPOUT = 0.1
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS # How big Query, Key and Value matrices are
EVAL_INTERVAL = 500
EVAL_LOSS_BATCHES = 200

device = 'cuda' if torch.cuda.is_available() else "cpu"