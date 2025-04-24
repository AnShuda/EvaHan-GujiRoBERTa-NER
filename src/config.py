MODEL_NAME = "hsc748NLP/GujiRoBERTa_jian_fan"

LABEL_LIST = [
    "O",
    "B-NR", "M-NR", "E-NR", "S-NR",
    "B-NS", "M-NS", "E-NS", "S-NS",
    "B-NB", "M-NB", "E-NB", "S-NB",
    "B-NO", "M-NO", "E-NO", "S-NO",
    "B-NG", "M-NG", "E-NG", "S-NG",
    "B-T",  "M-T",  "E-T",  "S-T",
    "B-ZD", "M-ZD", "E-ZD", "S-ZD",
    "B-ZZ", "M-ZZ", "E-ZZ", "S-ZZ",
    "B-ZF", "M-ZF", "E-ZF", "S-ZF",
    "B-ZP", "M-ZP", "E-ZP", "S-ZP",
    "B-ZS", "M-ZS", "E-ZS", "S-ZS",
    "B-ZA", "M-ZA", "E-ZA", "S-ZA",
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5