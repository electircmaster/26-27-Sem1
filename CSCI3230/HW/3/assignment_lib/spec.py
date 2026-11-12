import copy

SCHEMA = 'csci3230_a3_v5'
PAD, BOS, QUERY, SEP, EOS = range(5)
KEY_START, NUM_KEYS = 5, 16
VALUE_START, NUM_VALUES = 21, 8
INDEX_START, BAR, DOT = 29, 33, 34
VOCAB_SIZE = 35
TF_CONFIG = dict(vocab_size=VOCAB_SIZE, max_length=48, d_model=64,
                 num_heads=4, num_layers=2, d_ff=128)
DEFAULT_CONFIG = dict(schema=SCHEMA, cnn=dict(width=16), transformer=TF_CONFIG)
MAX_CNN_PARAMETERS = 200_000
MAX_WEIGHTS_BYTES = 8_000_000


def default_config():
    return copy.deepcopy(DEFAULT_CONFIG)

TF_TRAIN_STEPS = 300
TF_SEED = 3230
