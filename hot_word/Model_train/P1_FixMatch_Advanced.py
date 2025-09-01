import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv
from torch.utils.data import random_split
import yaml
import argparse
from hot_word.utils.config_loader import load_config
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import json
import re
import jieba


# === 配置解析 ===
def parse_args():
    parser = argparse.ArgumentParser(description="FixMatch Training with YAML configs")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--select", type=str, help="Select a specific config name in YAML (if multiple)")
    parser.add_argument("--lr", type=float, help="Override learning rate in YAML")
    parser.add_argument("--optimizer_type", type=str, help="Override optimizer")
    parser.add_argument("--nesterov", type=bool, help="Override nesterov for SGD")
    parser.add_argument("--weight_decay", type=float, help="Override weight_decay")
    parser.add_argument("--momentum", type=float, help="Override momentum")
    parser.add_argument("--scheduler", type=str, help="Override scheduler type")
    parser.add_argument("--model_path", type=str, help="Setting your model path")
    return parser.parse_args()


# === 增强函数 ===
def weak_augment(text):
    # 弱增强：随机删除一个词
    words = list(text)
    if len(words) > 10:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
    return "".join(words)


def strong_augment(text, strong_aug_ratio):
    # 强增强: 随机mask strong_aug_ratio%字符
    words = list(text)
    n_mask = max(1, int(len(words) * strong_aug_ratio))
    for _ in range(n_mask):
        idx = random.randint(0, len(words)-1)
        words[idx] = "[MASK]"
    return "".join(words)


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))


# 加载停用词表
def load_stopwords(stopwords_path=os.path.join(project_root, "hot_word", "Model_train", "hit_stopwords.txt")):
    stopwords = set()
    with open(stopwords_path, encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())
    return stopwords


stopwords = load_stopwords()


def clean_and_cut(text, stopwords=stopwords):
    # # 1. 去掉HTML标签
    text = re.sub(r"<.*?>", "", text)
    # # 2. 去掉URL
    text = re.sub(r"http[s]?://\S+", "", text)
    # # 3. 去掉标点和符号(只保留中英文和数字)
    text = re.sub(r"[^u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    # 4. 分词
    words = jieba.lcut(text)
    # 5. 去停用词 + 去掉过短的词
    words = [w for w in words if w not in stopwords and len(w.strip()) > 1]
    return " ".join(words)


# === 数据集 ===
class LabeledDataset(Dataset):
    def __init__(self, fold_path):
        self.samples = []
        for file in os.listdir(fold_path):
            if file.endswith(".csv"):
                file_path = os.path.join(fold_path, file)
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)  # 跳过表头
                    for i, parts in enumerate(reader):
                        try:
                            if len(parts) != 2:
                                print(f"Warning: line {i+2} does not have 2 parts:", parts)
                                continue
                            text, label = parts
                            text = clean_and_cut(text.strip())
                            if not text:
                                print(f"Warning: line {i+2} is empty text, skipped")
                                continue
                            self.samples.append((text, int(float(label))))
                        except Exception as e:
                            print(f"Error at line {i+2}: {parts}")
                            print("Exception: ", e)
                            continue

        # 打乱所有样本
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return text, label


class UnlabeledDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = []
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                with open(file_path, encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for i, parts in enumerate(reader):
                        try:
                            if len(parts) != 2:
                                print(f"Warning: line {i+2} does not have 2 parts:", parts)
                                continue
                            # 取第一列的文本列
                            text = clean_and_cut(parts[0].strip())  # 去掉首尾空格
                            if not text:
                                print(f"Warning: line {i+2} in file {file} is empty, skipped")
                                continue
                            self.samples.append(text)
                        except Exception as e:
                            print(f"error at line {i+2} in file {file}: {parts}")
                            continue

        # === 去重， 保持原顺序
        self.samples = list(dict.fromkeys(self.samples))
        # 打乱
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        return text


# 没有必要，因为全量采样加上高置信度一样可以起到退火作用，没必要限制这个每次采样大小。
def get_unlabeled_subset(full_dataset, ratio, labeled_size):
    target_size = int(labeled_size * ratio)
    target_size = min(target_size, len(full_dataset))
    indices = random.sample(range(len(full_dataset)), target_size)
    subset = torch.utils.data.Subset(full_dataset, indices)
    return subset


# === collate_fn ===
def collate_labeled(batch, tokenizer, max_len):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(labels)


def collate_unlabeled(batch, tokenizer, max_len):
    original_texts = batch  # 保留原始文本
    weak_texts = [weak_augment(t) for t in batch]
    # 这里只返回弱增强的tokenizer，强增强进行解耦后在训练步里面进行操作
    enc_w = tokenizer(weak_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc_w, original_texts


def collate_dev(batch, tokenizer, max_len):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(labels), texts


# === 模型 ===
# Hugging Face结构
class TextClassifier(nn.Module):
    def __init__(self, model_path, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = output.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits

    def set_dropout(self, p):
        self.dropout.p = p


# 混淆矩阵函数
def confusion_metrics(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # 计算混淆矩阵
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # 计算各类指标
    for i, label in enumerate(labels):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        acc = (TP + TN) / cm.sum() if cm.sum() > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Label {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    overall_acc = (y_true == y_pred).mean()

    return cm, overall_acc


# 传入loader即可对数据进行测试
def evaluate(model, loader, device, class_weights=None, print_report=False, info=None):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for enc, labels in loader:
            enc, labels = {k: v.to(device) for k, v in enc.items()}, labels.to(device)
            logits = model(**enc)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(all_labels)

    cm, overall_acc = confusion_metrics(all_labels, all_preds)
    recalls = {}
    for i, label in enumerate(np.unique(all_labels)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        recalls[label] = TP / (TP + FN) if (TP + FN) > 0 else 0

    R_pos = recalls.get(0, 0.0)  # 正类
    R_neg = recalls.get(1, 0.0)  # 负类
    s_recall = 2 * R_pos * R_neg / (R_pos + R_neg + 1e-8)

    if print_report:
        print(f"Confusion Matrix:{info}")
        print(cm)
        print(f"Overall Accuracy: {overall_acc:.4f}")

    return avg_loss, overall_acc, all_labels, all_preds, s_recall


# 评估Dev集，打印出错误样本
def evaluate_dev(model, loader, device, class_weights=None, print_report=False, info=None, save_mistakes_path=None, epoch=None, save_every=5, pos_threshold=0.5, neg_threshold=0.6):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    mistake_samples = []  # 存放错误样本

    with torch.no_grad():
        for enc, labels, texts in loader:
            enc, labels = {k: v.to(device) for k, v in enc.items()}, labels.to(device)
            logits = model(**enc)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item() * labels.size(0)

            # 推理阈值
            probs = F.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)

            # === 推理阈值逻辑 ===
            adjusted_preds = []
            for p, c in zip(preds, conf):
                if p.item() == 0 and c.item() < pos_threshold:
                    adjusted_preds.append(2)
                elif p.item() == 1 and c.item() < neg_threshold:
                    adjusted_preds.append(2)
                else:
                    adjusted_preds.append(p.item())
            preds = torch.tensor(adjusted_preds, device=labels.device)

            # 保存错误样本
            for t, gold, pred in zip(texts, labels.cpu().numpy(), preds.cpu().numpy()):
                if gold != pred:
                    mistake_samples.append({
                        "text": t,
                        "gold": int(gold),
                        "pred": int(pred)
                    })

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / len(all_labels)

    cm, overall_acc = confusion_metrics(all_labels, all_preds)
    recalls = {}
    for i, label in enumerate(np.unique(all_labels)):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        recalls[label] = TP / (TP + FN) if (TP + FN) > 0 else 0

    R_pos = recalls.get(0, 0.0)  # 正类
    R_neg = recalls.get(1, 0.0)  # 负类
    s_recall = 2 * R_pos * R_neg / (R_pos + R_neg + 1e-8)

    if print_report:
        print(f"Confusion Matrix:{info}")
        print(cm)
        print(f"Overall Accuracy: {overall_acc:.4f}")

    # 每隔一定epoch保存错误样本
    if save_mistakes_path and epoch is not None and epoch % save_every == 0:
        os.makedirs(save_mistakes_path, exist_ok=True)
        save_file = os.path.join(save_mistakes_path, f"mistakes_epoch{epoch}.jsonl")
        with open(save_file, "w", encoding="utf-8") as f:
            for item in mistake_samples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Info] Mistakes saved to {save_file}")

    return avg_loss, overall_acc, all_labels, all_preds, s_recall


# === EMA更新 ===
@torch.no_grad()
def ema_update(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)


# Dropout 增大
def set_dropout_p(model, p=0.5):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
    # 若是 HF Transformer, 更新 config
    if hasattr(model, "config"):
        if hasattr(model.config, "hidden_dropout_prob"):
            model.config.hidden_dropout_prob = p
        if hasattr(model.config, "attention_probs_dropout_prob"):
            model.config.attention_probs_dropout_prob = p


# class-balanced
def get_class_balanced_weights(n_samples_per_class, beta=0.999, boost_dict=None, device='cpu'):
    n_samples = np.array(n_samples_per_class, dtype=np.float32)
    # 计算 Effective Number
    eff_num = 1.0 - np.power(beta, n_samples)
    eff_num[eff_num == 0] = 1e-8
    weights = (1.0 - beta) / eff_num

    # 调试 boost_dict
    print(boost_dict)

    # 手动增强某些类
    if boost_dict is not None:
        for cls_idx, alpha in boost_dict.items():
            weights[int(cls_idx)] *= alpha

    # 均值归一化
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# R-Drop + Label Smoothing
def supervised_loss_rdrop_cb_ls(model, inputs, labels,
                                class_weights=None,
                                label_smoothing=0.05,
                                rdrop_alpha=0.5):
    # 两次前向(如果模型对同一个输入预测不太一致，就说明过拟合或者不稳定)
    logits1 = model(**inputs)
    logits2 = model(**inputs)

    # 1) CE with label smoothing + class balanced
    if class_weights is not None:
        ce1 = F.cross_entropy(logits1, labels, weight=class_weights, label_smoothing=label_smoothing)
        ce2 = F.cross_entropy(logits2, labels, weight=class_weights, label_smoothing=label_smoothing)
    else:
        ce1 = F.cross_entropy(logits1, labels, label_smoothing=label_smoothing)
        ce2 = F.cross_entropy(logits2, labels, label_smoothing=label_smoothing)

    ce_loss = 0.5 * (ce1 + ce2)

    # 2) 对称 KL
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    kl_loss = 0.5 * (F.kl_div(log_p1, p2, reduction="batchmean") +
                     F.kl_div(log_p2, p1, reduction="batchmean"))

    # 3)
    loss = ce_loss + rdrop_alpha * kl_loss
    return loss, ce_loss.detach(), kl_loss.detach()


# === FixMatch ===
def train(labeled_ds, unlabeled_ds, dev_ds, tokenizer, cfg, project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志路径
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{cfg.get('name', 'default')}"
    log_dir = os.path.join(project_root, cfg["logging"]["log_dir"], run_name)
    os.makedirs(log_dir, exist_ok=True)

    # 训练集和测试集
    total = len(labeled_ds)
    test_size = int(0.15 * total)
    train_size = total - test_size
    train_ds, test_ds = random_split(labeled_ds, [train_size, test_size])
    test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_l"], shuffle=False,
                             collate_fn=lambda b: collate_labeled(b, tokenizer, cfg["dataset"]["max_len"]))
    dev_loader = DataLoader(dev_ds, batch_size=cfg["training"]["batch_l"], shuffle=False,
                             collate_fn=lambda b: collate_dev(b, tokenizer, cfg["dataset"]["max_len"]))

    model = TextClassifier(cfg["model"]["model_path"]).to(device)
    ema_model = TextClassifier(cfg["model"]["model_path"]).to(device)
    ema_model.load_state_dict(model.state_dict())

    # --- class-balanced 权重 ---
    n_samples_per_class = cfg["dataset"]["n_samples_per_class"]
    boost_dict = cfg["dataset"]["boost_dict"]
    class_weights = get_class_balanced_weights(n_samples_per_class, beta=0.999,
                                              boost_dict=boost_dict, device=device)

    # 优化器选择
    optimizer_type = cfg["training"]["optimizer_type"].lower()
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    elif optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg["training"]["lr"],
                              momentum=cfg["training"]["momentum"], weight_decay=cfg["training"]["weight_decay"],
                              nesterov=cfg["training"]["nesterov"])
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # 学习率调度器
    scheduler_type = cfg["training"]["scheduler"].lower()
    if scheduler_type == "cosine":
        # 学习率衰减函数
        total_steps = cfg["training"]["epochs"] * (len(train_ds) // cfg["training"]["batch_l"])  # 总训练步

        def lr_lambda(current_step):
            # 余弦衰减:
            # n * cosine(7πk/16K)
            return math.cos((7 * math.pi * current_step) / (16 * total_steps))

        scheduler = LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    scaler = GradScaler(enabled=cfg["training"]["use_amp"])

    labeled_size = len(train_ds)
    full_unlabeled_size = len(unlabeled_ds)

    print(labeled_size)
    print(full_unlabeled_size)

    # 梯度积累模拟大BATCH
    physical_B_U = 64  # 实际送GPU的无标签batch大小
    accum_steps_u = cfg["training"]["batch_u"] // physical_B_U
    # accum_counter = 0

    warmup_epochs = cfg["training"]["warmup_epochs"]
    best_s_recall = 0.0
    best_ema_acc = 0.0
    best_model_dir = os.path.join(project_root, cfg["model"]["best_model_dir"])
    os.makedirs(best_model_dir, exist_ok=True)
    global_step = 0
    collapse_threshold = cfg["training"].get("collapse_threshold", 0.95)

    loader_l = DataLoader(train_ds, batch_size=cfg["training"]["batch_l"], shuffle=True,
                          collate_fn=lambda b: collate_labeled(b, tokenizer, cfg["dataset"]["max_len"]))
    loader_u = DataLoader(unlabeled_ds, batch_size=physical_B_U, shuffle=True,
                          collate_fn=lambda b: collate_unlabeled(b, tokenizer, cfg["dataset"]["max_len"]))
    it_u = iter(loader_u)

    # === EMA 监控初始化 ===
    num_classes = cfg["model"]["num_labels"]
    ema_alpha = 0.9
    selected_fraction_c_ema = torch.zeros(num_classes, device=device)
    loss_u_contrib_c_ema = torch.zeros(num_classes, device=device)

    # 保存路径
    best_model_path_frozen_model = os.path.join(best_model_dir, "fixmatch_model_frozen_model.pt")
    best_model_path_frozen_ema_model = os.path.join(best_model_dir, "fixmatch_model_frozen_ema_model.pt")
    best_model_path_semi_model = os.path.join(best_model_dir, "fixmatch_model_semi_model.pt")
    best_model_path_semi_ema_model = os.path.join(best_model_dir, "fixmatch_model_semi_ema_model.pt")

    for epoch in range(cfg["training"]["epochs"]):
        k = epoch + 1
        # 计算无标签采样比例
        # alpha = 1
        # ratio = (alpha * k / cfg["training"]["epochs"]) * full_unlabeled_size / labeled_size
        # ratio = min(ratio, full_unlabeled_size / labeled_size)

        print(f"=== Epoch {k}, 使用全量无标签数据 ===")
        # unlabeled_subset = get_unlabeled_subset(unlabeled_ds, ratio, labeled_size)

        # loader_u = DataLoader(unlabeled_subset, batch_size=physical_B_U, shuffle=True,
        #                       collate_fn=lambda b: collate_unlabeled(b, tokenizer, cfg["dataset"]["max_len"]))
        # it_u = iter(loader_u)

        # === 冻结阶段判断 ===
        if epoch < warmup_epochs:
            print(f"Epoch {epoch+1}: WARM-UP PHASE")
            lambda_u = 0.0
            # 冻结底层
            for param in model.encoder.parameters():
                param.requires_grad = False
            # dropout 调大
            model.set_dropout(0.5)
        else:
            # 解冻第一轮，加载冻结阶段最佳模型
            if epoch == warmup_epochs:
                if os.path.exists(best_model_path_frozen_model) and os.path.exists(best_model_path_frozen_ema_model):
                    model.load_state_dict(torch.load(best_model_path_frozen_model, map_location=device))
                    ema_model.load_state_dict(torch.load(best_model_path_frozen_ema_model, map_location=device))
                    print(f"[Init Semi] 已加载冻结阶段最佳模型作为解冻起点 (acc={best_ema_acc:.4f})")
                else:
                    print("[Init Semi] 未找到冻结阶段最佳模型，继续使用当前参数")
            print(f"Epoch {epoch+1}: SEMI-SUPERVISED PHASE")
            lambda_u = cfg["training"]["lambda_u"]
            for param in model.encoder.parameters():
                param.requires_grad = True
            model.set_dropout(0.3)

        for enc_l, y_l in loader_l:
            # 有标签数据移动到GPU
            enc_l, y_l = {k: v.to(device) for k, v in enc_l.items()}, y_l.to(device)

            with autocast(enabled=cfg["training"]["use_amp"]):
                # 有标签损失
                loss_l, ce_loss, kl_loss = supervised_loss_rdrop_cb_ls(
                    model,
                    enc_l,
                    y_l,
                    class_weights=class_weights,
                    label_smoothing=0.05,
                    rdrop_alpha=0.5
                )

                # 累加无标签损失
                loss_u_total = 0.0
                # 非冻结时候才引入无标签训练
                if lambda_u > 0:
                    for _ in range(accum_steps_u):
                        # 取无标签batch
                        try:
                            enc_u_w, orig_u_texts = next(it_u)
                        except StopIteration:
                            it_u = iter(loader_u)
                            enc_u_w, orig_u_texts = next(it_u)

                        # 无标签数据移动到GPU
                        enc_u_w = {k: v.to(device) for k, v in enc_u_w.items()}

                        # 无标签弱增强->伪标签,进一步对每一类预测进行各自的强增强
                        with torch.no_grad():
                            logits_u_w = ema_model(**enc_u_w)
                            probs_u_w = torch.softmax(logits_u_w, dim=-1)
                            max_probs, pseudo_labels = torch.max(probs_u_w, dim=-1)
                            # 打印调试阈值
                            pmax = probs_u_w.max(dim=1).values

                            IGNORE_INDEX = -1
                            num_labeled_in_batch = int((y_l.long() != IGNORE_INDEX).sum())

                            # === 类自适应强增强和mask构建 ===
                            strong_texts = []
                            mask = torch.zeros(len(orig_u_texts), device=device)  # mask 初始化
                            weighted_cls = torch.full((len(orig_u_texts),), -1, dtype=torch.long, device=device)  # 全-1列表，用于进行类权重计算

                            for i, text in enumerate(orig_u_texts):
                                cls = pseudo_labels[i].item()
                                threshold = cfg["training"]["thresholds"][cls]  # 类自适应阈值
                                aug_ratio = cfg["training"]["strong_aug_ratio"][cls]  # 类自适应增强比例

                                if max_probs[i] >= threshold:
                                    # 每类高置信度样本参与强增强
                                    strong_texts.append(strong_augment(text, aug_ratio))
                                    mask[i] = 1.0
                                    weighted_cls[i] = cls  # 记录参与增强的类别
                                else:
                                    # 低置信样本不增强，为了loss统一，直接append原文
                                    strong_texts.append(text)

                            # tokenizer
                            enc_u_s = tokenizer(strong_texts, padding=True, truncation=True, max_length=cfg["dataset"]["max_len"], return_tensors="pt")
                            enc_u_s = {k: v.to(device) for k, v in enc_u_s.items()}

                            # 基于 weighted_cls 计算类别权重(先计数用于伪标签类配额检查，再归一化)
                            weights_pseudolabel = torch.zeros(num_classes, device=device)
                            for c in range(num_classes):
                                weights_pseudolabel[c] = (weighted_cls == c).sum().float()

                            total_selected = weights_pseudolabel.sum()

                            if total_selected.item() > 0:
                                class_counts = weights_pseudolabel.clone()  # 每类被选中的数量
                                weights_pseudolabel = class_counts / total_selected  # 归一化
                            else:
                                class_counts = weights_pseudolabel.clone()
                                weights_pseudolabel = torch.ones(num_classes, device=device) / num_classes  # fallback 防止除0

                            # 读取每类标签
                            pos_id = cfg["training"]["pos_id"]
                            neg_id = cfg["training"]["neg_id"]
                            neutral_id = cfg["training"]["neutral_id"]
                            # === 基于入选样本的伪标签类配额进行质量控制 ===
                            if total_selected.item() > 0:
                                total = total_selected.item()
                                pos_ratio = class_counts[pos_id].item() / total
                                neg_ratio = class_counts[neg_id].item() / total
                                neutral_ratio = class_counts[neutral_id].item() / total

                                # 判定规则
                                neutral_max = cfg["training"].get("neutral_max_ratio", 0.7)
                                if neutral_ratio > neutral_max or neutral_ratio > 2 * (pos_ratio + neg_ratio):
                                    # 整批丢弃
                                    continue

                            # 强增强一致性损失
                            logits_u_s = model(**enc_u_s)
                            loss_u_per_sample = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none', weight=weights_pseudolabel)
                            # loss_u_per_sample 中有低置信度的样本，要丢弃
                            loss_u = (loss_u_per_sample * mask).mean()
                            loss_u_total += loss_u  # 直接进行累加，不除以7

                            # === 监控 Selected Fraction_c & Loss_u Contribution_c + EMA ===
                            selected_fraction_c = torch.zeros(num_classes, device=device)
                            loss_u_contrib_c = torch.zeros(num_classes, device=device)

                            for c in range(num_classes):
                                mask_c = (pseudo_labels == c)
                                if mask_c.sum() > 0:
                                    selected_fraction_c[c] = (mask[mask_c].sum() / (mask_c.sum() + 1e-8))
                                    loss_u_contrib_c[c] = (loss_u_per_sample[mask_c].sum() / (loss_u_per_sample.sum() + 1e-8))

                            # EMA 平滑
                            selected_fraction_c_ema = ema_alpha * selected_fraction_c_ema + (1 - ema_alpha) * selected_fraction_c
                            loss_u_contrib_c_ema = ema_alpha * loss_u_contrib_c_ema + (1 - ema_alpha) * loss_u_contrib_c

                            # 隔步打印
                            if global_step % cfg["training"].get("log_every_steps", 500) == 0:
                                print(f"[Monitor][Step {global_step}] Selected Fraction EMA: {selected_fraction_c_ema.tolist()}")
                                print(f"[Monitor][Step {global_step}] Loss_u Contribution EMA: {loss_u_contrib_c_ema.tolist()}")

            # 总损失
            loss = loss_l + lambda_u * loss_u_total

            # 清空梯度
            optimizer.zero_grad()

            # backward
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            ema_update(ema_model, model, cfg["training"]["ema_decay"])

            global_step += 1

        # === 每个epoch验证 ===
        ema_loss_test, ema_accuracy_test, _, _, ema_s_recall_test = evaluate(ema_model, test_loader, device, class_weights=None, print_report=True, info="ema_test")
        ema_loss_dev, ema_accuracy_dev, _, all_preds, ema_s_recall_dev = evaluate_dev(ema_model, dev_loader, device, class_weights=None, print_report=True, info="ema_dev", save_mistakes_path=cfg["logging"]["save_mistakes_path"], epoch=epoch+1)

        # 预测分布监控(进阶各类式)
        counts = np.bincount(all_preds, minlength=cfg["model"]["num_labels"])
        probs = counts / counts.sum()
        print(f"[Monitor] ema Dev prediction distribution: {probs}")

        # 检查坍缩 (只在半监督阶段生效)
        if lambda_u > 0 and probs.max() > collapse_threshold:
            print(f"[Warning] Collapse detected at epoch {epoch+1} !"
                  f"One class 占比 {probs.max():.2f} > {collapse_threshold}. 回溯至checkpoint")
            if os.path.exists(best_model_path_semi_model) and os.path.exists(best_model_path_semi_ema_model):
                model.load_state_dict(torch.load(best_model_path_semi_model))
                ema_model.load_state_dict(torch.load(best_model_path_semi_ema_model))
                print(f"[Recovery] 模型已回滚到最佳 dev checkpoint (acc={ema_accuracy_dev:.4f}")
            else:
                print("[Recovery] 没有找到已保存的最佳模型，跳过")

        print(f"Epoch {epoch+1} loss: {ema_loss_dev:.4f}, acc: {ema_accuracy_dev:.4f}")

        # 仅在性能更好的时候(正负召回)才保存checkpoint(只使用ema模型)
        # === 分阶段保存逻辑 ===
        if lambda_u == 0.0:  # 冻结阶段
            if ema_accuracy_test > best_ema_acc:
                best_ema_acc = ema_accuracy_test
                torch.save(model.state_dict(), best_model_path_frozen_model)
                torch.save(ema_model.state_dict(), best_model_path_frozen_ema_model)
                print(f"[Warm-up] 保存冻结阶段最佳模型 (acc={best_ema_acc:.4f})")
            else:  # 如果模型效果没有更好
                if os.path.exists(best_model_path_frozen_model) and os.path.exists(best_model_path_frozen_ema_model):
                    model.load_state_dict(torch.load(best_model_path_frozen_model))
                    ema_model.load_state_dict(torch.load(best_model_path_frozen_ema_model))
                    print(f"[Warm-up] 精度未提升，已回退至最佳模型 (acc={best_ema_acc:.4f})")
                else:
                    print("[Warm-up] 没有找到已保存的最佳模型，保持当前参数")
        else:  # 半监督阶段
            if ema_s_recall_dev > best_s_recall:
                best_s_recall = ema_s_recall_dev
                torch.save(model.state_dict(), best_model_path_semi_model)
                torch.save(ema_model.state_dict(), best_model_path_semi_ema_model)
                print(f"Saved new best EMA model at epoch {k} (acc={ema_accuracy_dev:.4f})")

    print("Training done")
    # writer.close()


if __name__ == "__main__":
    args = parse_args()
    configs = load_config(args.config, args.select)

    for cfg in configs:
        # 命令行覆盖YAML参数
        if args.lr is not None:
            cfg["training"]["lr"] = args.lr
        if args.optimizer_type is not None:
            cfg["training"]["optimizer_type"] = args.optimizer_type
        if args.nesterov is not None:
            cfg["training"]["nesterov"] = args.nesterov
        if args.weight_decay is not None:
            cfg["training"]["weight_decay"] = args.weight_decay
        if args.momentum is not None:
            cfg["training"]["momentum"] = args.momentum
        if args.scheduler is not None:
            cfg["training"]["scheduler"] = args.scheduler
        if args.model_path is not None:
            cfg["training"]["model_path"] = args.model_path

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_root, cfg["model"]["model_path"]))
        labeled_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["labeled_path"]))
        unlabeled_ds = UnlabeledDataset(os.path.join(project_root, cfg["dataset"]["unlabeled_path"]))
        dev_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["dev_path"]))

        print(f"===== Running config: {cfg.get('name', 'default')} =====")
        train(labeled_ds, unlabeled_ds, dev_ds, tokenizer, cfg, project_root)

    print("end")
