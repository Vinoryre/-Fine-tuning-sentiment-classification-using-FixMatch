from transformers import BertTokenizer, BertModel
import torch
import os

# 当前脚本所在目录,以及根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
# print("项目根目录: ", project_root)

# 模型本地路径
model_path = os.path.join(project_root, "model_test", "chinese_roberta_wwm_ext")
print(model_path)

# 加载分词器还有模型
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

text = "今天天气真好，我们去游泳吧"
inputs = tokenizer(text, return_tensors="pt")

# 前向计算
with torch.no_grad():
    outputs = model(**inputs)

# 取[CLS]向量
cls_embeddding = outputs.last_hidden_state[:, 0, :]
print("CLS向量形状： ", cls_embeddding.shape)
print("CLS向量数据(top5): ", cls_embeddding[0][:5])
print("end")