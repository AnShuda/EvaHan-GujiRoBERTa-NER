from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.nn.functional import softmax
import torch
import os

MAX_LEN = 510
model_path = "models/roberta_model/checkpoint-3099"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()

input_base_path = "data/TestSet/sequence"
output_base_path = "predictions"
os.makedirs(output_base_path, exist_ok=True)

for set_name in ["A", "B", "C"]:
    input_file = os.path.join(input_base_path, f"testset_{set_name}.txt")
    output_file = os.path.join(output_base_path, f"testset_{set_name}_pred.txt")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 所有字汇总成一整段（忽略空行）
    chars = [line.strip() for line in lines if line.strip()]
    output_lines = []

    for i in range(0, len(chars), MAX_LEN):
        sub_chars = chars[i:i+MAX_LEN]
        inputs = tokenizer(sub_chars, is_split_into_words=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()
        labels = [model.config.id2label[p] for p in predictions[1:len(sub_chars)+1]]

        for char, label in zip(sub_chars, labels):
            output_lines.append(f"{char}\t{label}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"✅ testset_{set_name} 预测完成，结果已保存到 {output_file}")