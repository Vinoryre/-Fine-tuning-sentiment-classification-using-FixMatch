import os
import random
import pandas as pd
import nlpaug.augmenter.word as naw
import nltk
import jieba
import nlpaug.augmenter.sentence as nas
from tqdm import tqdm

nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download("omw-1.4")
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(current_path))
# 配置
ORIG_FOLDER = os.path.join(project_path, "hot_word", "Dataset", "F1_data", "labeled")
SAVE_FOLDER = os.path.join(project_path, "hot_word", "Dataset", "F1_data", "labeled_aug")
SEED = 42
random.seed(SEED)

# 增强器
AUGS = [
    naw.RandomWordAug(action='swap'),
    naw.RandomWordAug(action='delete'),
]


def augment_text_folder(input_folder, output_folder, augment_times=2):
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for file_name in all_files:
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        new_rows = []

        for idx, row in df.iterrows():
            text = row['text']
            label = row['label']

            # 正负类增强
            if label in [0, 1]:
                aug = random.choice(AUGS)
                tokens = list(jieba.cut(text))
                aug_text = ' '.join(tokens)
                for _ in range(augment_times):
                    aug_text = aug.augment(aug_text)
                aug_text = aug_text.replace(' ', '')
                new_rows.append({'text': aug_text, 'label': label})
            else:
                # 中性类直接保留
                new_rows.append({'text': text, 'label': label})

            # 原始样本保留
            new_rows.append({'text': text, 'label': label})

        out_file = os.path.join(output_folder, f"aug_{file_name}")
        pd.DataFrame(new_rows).to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f"[Done] {file_name} -> {out_file}, total rows: {len(new_rows)}")


augment_text_folder(ORIG_FOLDER, SAVE_FOLDER)
