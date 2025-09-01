import pandas as pd
import glob

# 读取xlsx 文件
# for xlsx_file in glob.glob("../Dataset/P1_data/labeled/*.xlsx"):
#     df = pd.read_excel(xlsx_file, engine='openpyxl')
#     csv_file = xlsx_file.replace(".xlsx", ".csv")
#     df.to_csv(csv_file, index=False, encoding='utf-8-sig')
#     print("finish!")

# 正向修正
df = pd.read_excel("../Dataset/P1_data/labeled_10_65_100/labeled_positive.xlsx", engine='openpyxl')
keep_cols = ["chat_content", "tap"]
df = df[keep_cols]
df = df.rename(columns={
    "chat_content": "text",
    "tap": "label"
})
df["label"] = 0
df = df.sample(n=10000, random_state=42)
df.to_csv("../Dataset/P1_data/labeled_10_65_100/labeled_positive.csv", index=False, encoding="utf-8-sig")
print("positive FINISH")

# 负向修正
df = pd.read_excel("../Dataset/P1_data/labeled_10_65_100/labeled_negative.xlsx", engine='openpyxl')
df = df[keep_cols]
df = df.rename(columns={
    "chat_content": "text",
    "tap": "label"
})
df["label"] = 1
df = df.sample(n=65000, random_state=42)
df.to_csv("../Dataset/P1_data/labeled_10_65_100/labeled_negative.csv", index=False, encoding="utf-8-sig")
print("negative FINISH")

# 中性修正
df = pd.read_excel("../Dataset/P1_data/labeled_10_65_100/labeled_neutral.xlsx", engine='openpyxl')
keep_cols = ["chat_content", "model_score1"]
df = df[keep_cols]
df = df.rename(columns={
    "chat_content": "text",
    "model_score1": "label"
})
df["label"] = 2
n = min(100000, len(df))
df = df.sample(n=n, random_state=42)
df.to_csv("../Dataset/P1_data/labeled_10_65_100/labeled_neutral.csv", index=False, encoding="utf-8-sig")
print("neutral FINISH")

print("end")