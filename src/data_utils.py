def load_bio_file(file_path):
    '''
    输入路径：标注格式文本文件路径（支持 BIO、BMEOS 等）
    输出：
        - token_list: List[List[str]] 每个句子的字列表
        - label_list: List[List[str]] 每个句子的标签列表
    '''
    token_list = []
    label_list = []
    current_tokens = []
    current_labels = []
    sentence_end_punct = {'。', '！', '？', '；'}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    token_list.append(current_tokens)
                    label_list.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue
            try:
                token, label = line.split(None, 1)
            except ValueError:
                continue
            current_tokens.append(token)
            current_labels.append(label)

            if token in sentence_end_punct:
                token_list.append(current_tokens)
                label_list.append(current_labels)
                current_tokens = []
                current_labels = []

        # 文件末尾处理
        if current_tokens:
            token_list.append(current_tokens)
            label_list.append(current_labels)

    print(f"共加载了 {len(token_list)} 个样本")
    return token_list, label_list