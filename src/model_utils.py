from transformers import AutoTokenizer, AutoModelForTokenClassification
from config import MODEL_NAME

def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)

def load_model(num_labels):
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )