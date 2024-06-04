import torch

# Hyperparameters
NUM_MERGES = 10000
SPECIAL_TOKENS = [b"<|STARTOFTEXT|>", b"<|ENDOFTEXT|>", b"<|PAD|>"]
VOCAB_SIZE = 256 + NUM_MERGES + len(SPECIAL_TOKENS)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Increased sizes for better model capacity
EMBEDDING_SIZE = 512
CONTEXT_SIZE = 256
BATCH_SIZE = 64  # Increased batch size for GPU
MAX_STEPS = 50000  # More steps for better training
LEARNING_RATE = 3E-4  # Slightly reduced learning rate for stability
ENCODER_BLOCK_COUNT = 6  # More layers for deeper model
DECODER_BLOCK_COUNT = 6
NUM_HEADS = 8  # More heads for finer attention
DROPOUT = 0.1
HEAD_SIZE = EMBEDDING_SIZE // NUM_HEADS  # How big Query, Key and Value matrices are
EVAL_INTERVAL = 1000  # Evaluate less frequently with more training steps
EVAL_LOSS_BATCHES = 200

device = 'cuda' if torch.cuda.is_available() else "cpu"