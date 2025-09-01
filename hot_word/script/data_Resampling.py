import os
import random
import pandas as pd
import jieba
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(os.path.dirname(current_path))
# 配置
ORIG_FOLDER = os.path.join(project_path, "hot_word", "Dataset", "P1_data", "labeled_841")
SAVE_FOLDER = os.path.join(project_path, "hot_word", "Dataset", "P1_data", "labeled_841_Resampling")
SEED = 42
random.seed(SEED)


def resampling_text_folder(input_folder, output_folder, re_time_neg=1, re_time_neu=3):
    all_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for file_name in all_files:
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        new_rows = []

        for idx, row in df.iterrows():
            text = row['text']
            label = row['label']

            # 负类重采样一次
            if label in [1]:
                for _ in range(re_time_neg):
                    new_rows.append({'text': text, 'label': label})
            # 正类无需重采样
            elif label in [0]:
                pass
            # 中性重采样三次
            else:
                for _ in range(re_time_neu):
                    new_rows.append({'text': text, 'label': label})
            # 原始样本保留
            new_rows.append({'text': text, 'label': label})

        out_file = os.path.join(output_folder, f"Resampling_{file_name}")
        pd.DataFrame(new_rows).to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f"[Done] {file_name} -> {out_file}, total rows: {len(new_rows)}")


resampling_text_folder(ORIG_FOLDER, SAVE_FOLDER)
