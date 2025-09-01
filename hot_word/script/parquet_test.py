import pandas as pd
import os

def read_parquet(file_path: str):
    try:
        df = pd.read_parquet(file_path)
        print(f"文件读取成功")
        print(df.head())
        return df
    except Exception as e:
        print(f"读取 parquet 文件失败: {e}")
        return None


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(os.path.dirname(current_path))
    file_path = os.path.join(project_path, "hot_word", "Dataset", "imdb", "train.parquet")
    df = read_parquet(file_path)
