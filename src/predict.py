from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from config import LABEL_LIST, ID2LABEL
import argparse

def align_predictions(predictions, word_ids):
    preds = predictions.argmax(dim=-1).squeeze().tolist()
    aligned = []
    previous_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # 只对每个字第一次出现的 token 打标签
        if word_idx != previous_word_idx:
            aligned.append(ID2LABEL[preds[idx]])
            previous_word_idx = word_idx

    return aligned

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model_path', type=str, default="./models/roberta_model/checkpoint-3099")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    model.eval()

    text_chars = list(args.text)
    encodings = tokenizer(text_chars, 
                          is_split_into_words=True,
                          return_offsets_mapping=True,
                          return_tensors="pt",
                          truncation=True,
                          padding='max_length',
                          max_length=128)
    
    if 'offset_mapping' in encodings:
        encodings.pop('offset_mapping')

    with torch.no_grad():
        outputs = model(**encodings)
    
    word_ids = encodings.word_ids()
    predictions = align_predictions(outputs.logits, word_ids)

    print(f"\n输入文本：{args.text}")
    print("识别结果：")
    for char, label in zip(text_chars, predictions):
        print(f"{char} → {label}")

if __name__ == "__main__":
    main()