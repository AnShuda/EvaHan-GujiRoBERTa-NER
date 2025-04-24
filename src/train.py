from transformers import Trainer, TrainingArguments
from model_utils import load_tokenizer, load_model
from data_utils import load_bio_file
from dataset import NERDataset
from config import LABEL_LIST, LABEL2ID, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE

from sklearn.model_selection import train_test_split

# === 加载数据 ===
tokens_A, labels_A = load_bio_file("data/EvaHan2025_traingdata/trainset_A.txt")
tokens_B, labels_B = load_bio_file("data/EvaHan2025_traingdata/trainset_B.txt")
tokens_C, labels_C = load_bio_file("data/EvaHan2025_traingdata/trainset_C.txt")
tokens = tokens_A + tokens_B + tokens_C
labels = labels_A + labels_B + labels_C

# === 分词 & 对齐标签 ===
tokenizer = load_tokenizer()

def encode(tags, texts):
    encodings = tokenizer(texts, is_split_into_words=True,
                          truncation=True, padding='max_length',
                          max_length=MAX_LEN)
    all_labels = []
    for i, label in enumerate(tags):
        word_ids = encodings.word_ids(batch_index=i)
        label_ids = []
        prev = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev:
                label_ids.append(LABEL2ID.get(label[word_id], 0))
            else:
                label_ids.append(LABEL2ID.get(label[word_id], 0))
            prev = word_id
        all_labels.append(label_ids)
    return encodings, all_labels

X_train, X_val, y_train, y_val = train_test_split(tokens, labels, test_size=0.1)
train_enc, train_lab = encode(y_train, X_train)
val_enc, val_lab = encode(y_val, X_val)

train_dataset = NERDataset(train_enc, train_lab)
val_dataset = NERDataset(val_enc, val_lab)

# === 初始化模型 ===
model = load_model(num_labels=len(LABEL_LIST))

# === 训练参数 ===
args = TrainingArguments(
    output_dir='./models/roberta_model',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()