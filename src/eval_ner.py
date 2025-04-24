import os

def parse_ner_file(file_path):
    sentences = []
    current = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.strip().split()
                if len(parts) == 2:
                    current.append((parts[0], parts[1]))
        if current:
            sentences.append(current)
    return sentences

def extract_entities(sentence):
    entities = []
    i = 0
    while i < len(sentence):
        char, tag = sentence[i]
        if tag.startswith("B-"):
            label = tag[2:]
            start = i
            i += 1
            while i < len(sentence) and sentence[i][1] == f"I-{label}":
                i += 1
            end = i - 1
            entities.append((start, end, label))
        elif tag.startswith("S-"):
            label = tag[2:]
            entities.append((i, i, label))
            i += 1
        else:
            i += 1
    return entities

def evaluate(gold_path, pred_path):
    gold_data = parse_ner_file(gold_path)
    pred_data = parse_ner_file(pred_path)
    assert len(gold_data) == len(pred_data), "句子数不一致"

    gold_entities = []
    pred_entities = []

    for g, p in zip(gold_data, pred_data):
        assert len(g) == len(p), "句子长度不一致"
        gold_entities.extend(extract_entities(g))
        pred_entities.extend(extract_entities(p))

    gold_set = set(gold_entities)
    pred_set = set(pred_entities)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1": round(f1 * 100, 2)
    }

# === 多数据集评估 ===
sets = ["A", "B", "C"]
gold_base = "data/TestSet/label"
pred_base = "predictions"

macro_f1s = []

for s in sets:
    gold_path = os.path.join(gold_base, f"testset_{s}_raw_pred.txt")
    pred_path = os.path.join(pred_base, f"testset_{s}_pred.txt")
    if not os.path.exists(gold_path) or not os.path.exists(pred_path):
        print(f"❌ 缺失文件：testset_{s}")
        continue
    results = evaluate(gold_path, pred_path)
    macro_f1s.append(results["F1"])
    print(f"\n📊 testset_{s} 评估结果：")
    for k, v in results.items():
        print(f"{k}: {v}")

if macro_f1s:
    avg_f1 = sum(macro_f1s) / len(macro_f1s)
    print(f"\n⭐ Macro-F1 平均分（三个数据集）: {round(avg_f1, 2)}")