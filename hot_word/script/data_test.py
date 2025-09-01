from collections import Counter
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#df_0218_all_0205_final.csv
df = pd.read_csv("../Dataset/P1_data/Dev/labeled.csv")
label_counts = Counter(df['label'])
print(label_counts)

# 0205_final.csv
df = pd.read_csv("../Dataset/Manual_verification_set/golden_dataset.csv")
label_counts = Counter(df['label'])
print(label_counts)

# 抽样检测ai标注数据集准确率
df_true = pd.read_csv('../Dataset/Manual_verification_set/ai_0205_final_278_Manual.csv')
df_pred = pd.read_csv('../Dataset/Manual_verification_set/ai_0205_final_384_annotate.csv')

assert len(df_true) == len(df_pred), "行数不一致，请检查"

y_true = df_true['label'].astype(int)
y_pred = pd.to_numeric(df_pred['label'], errors='coerce').fillna(-1).astype(int)

accuracy = accuracy_score(y_true, y_pred)
print(f"准确率: {accuracy:.4f}")

cm = confusion_matrix(y_true, y_pred)
print("混淆矩阵: ")
print(cm)

print("分类报告: ")
print(classification_report(y_true, y_pred))

print('end')
