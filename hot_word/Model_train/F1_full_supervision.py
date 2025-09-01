import json
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import csv
import argparse
from hot_word.utils.config_loader import load_config
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np


# === 配置解析 ===
def parse_args():
    parser = argparse.ArgumentParser(description="Supervision Training with YAML configs")
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


# 终端更新配置函数
def update_config_with_args(cfg, args):
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
        cfg["model"]["model_path"] = args.model_path

    return cfg


# Dataset,train,test,dev
class LabeledDataset(Dataset):
    def __init__(self, fold_path):
        self.samples = []
        for file in os.listdir(fold_path):
            if file.endswith(".csv"):
                with open(os.path.join(fold_path, file), encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for parts in reader:
                        if len(parts) != 2:
                            continue
                        text, label = parts
                        text = text.strip()
                        if text:
                            self.samples.append((text, int(float(label))))
        random.shuffle(self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# 张量处理
def collate_labeled(batch, tokenizer, max_len):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(labels)


def collate_dev(batch, tokenizer, max_len):
    texts, labels = zip(*batch)
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(labels), texts


# 分类器
class TextClassifier(nn.Module):
    def __init__(self, model_path, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_path)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(pooled)


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


# 评估测试集(2:2:1)
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


# 评估Dev集,打印出错误样本
def evaluate_dev(model, loader, device, class_weights=None, print_report=False, info=None, save_mistakes_path=None, epoch=None, save_every=5):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    mistake_smaples = [] # 存放错误样本

    with torch.no_grad():
        for enc, labels, texts in loader:
            enc, labels = {k: v.to(device) for k, v in enc.items()}, labels.to(device)
            logits = model(**enc)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            total_loss += loss.item() * labels.size(0)

            preds = logits.argmax(dim=-1)

            # 保存错误样本
            for t, gold, pred in zip(texts, labels.cpu().numpy(), preds.cpu().numpy()):
                if gold != pred:
                    mistake_smaples.append({
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
            for item in mistake_smaples:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"[Info] Mistakes saved to {save_file}")

    return avg_loss, overall_acc, all_labels, all_preds, s_recall


# 训练
def train_supervised(labeled_ds, dev_ds, tokenizer, cfg, project_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total = len(labeled_ds)
    val_size = int(0.15 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(labeled_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_l"], shuffle=True,
                              collate_fn=lambda b: collate_labeled(b, tokenizer, cfg["dataset"]["max_len"]))
    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_l"], shuffle=False,
                            collate_fn=lambda b: collate_labeled(b, tokenizer, cfg["dataset"]["max_len"]))
    dev_loader = DataLoader(dev_ds, batch_size=cfg["training"]["batch_d"], shuffle=False,
                            collate_fn=lambda b: collate_dev(b, tokenizer, cfg["dataset"]["max_len"]))

    model = TextClassifier(os.path.join(project_root, cfg["model"]["model_path"]), num_labels=cfg["model"]["num_labels"]).to(device)

    # optimizer
    optimizer_type = cfg["training"]["optimizer_type"].lower()
    if optimizer_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg["training"]["lr"],
                              momentum=cfg["training"].get("momentum", 0.9),
                              weight_decay=cfg["training"]["weight_decay"],
                              nesterov=cfg["training"].get("nesterov", False))

    scheduler = None
    if cfg["training"].get("scheduler", "none").lower() == "cosine":
        steps_per_epoch = math.ceil(len(train_ds) / cfg["training"]["batch_l"])
        total_steps = max(1, cfg["training"]["epoch"]) * steps_per_epoch
        scheduler = LambdaLR(optimizer, lambda step: math.cos((7 * math.pi * step) / (16 * total_steps)))

    scaler = GradScaler(enabled=cfg["training"]["use_amp"])
    #writer = SummaryWriter(log_dir=os.path.join(project_root, cfg["logging"]["log_dir"]))

    best_acc = 0
    best_s_recall = 0
    best_model_path = os.path.join(project_root, cfg["model"]["best_model_path"])
    global_step = 0

    collapse_threshold = cfg["training"].get("collage_threshold", 0.95)  # 95%的坍缩阈值

    # === 类别权重 ===(自定义)
    class_counts = np.bincount([label for _, label in labeled_ds])
    custom_weight = np.ones_like(class_counts, dtype=np.float32)
    custom_weight[0] = 2.0  # 正类
    custom_weight[1] = 8.0  # 负类
    custom_weight[2] = 1.0  # 中性

    # 归一化
    custom_weight = custom_weight / custom_weight.sum() * len(custom_weight)

    class_weights = torch.tensor(custom_weight, dtype=torch.float).to(device)

    print(f"[Info] Using class weights: {class_weights}")

    for epoch in range(cfg["training"]["epoch"]):
        model.train()
        for enc, labels in train_loader:
            enc, labels = {k: v.to(device) for k, v in enc.items()}, labels.to(device)
            with autocast(enabled=cfg["training"]["use_amp"]):
                logits = model(**enc)
                loss = F.cross_entropy(logits, labels, weight=class_weights)

            # AMP backward -> grad clip -> optimizer step
            scaler.scale(loss).backward()

            # unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # scheduler step AFTER optimizer.step
            if scheduler:
                scheduler.step()

            optimizer.zero_grad()

            #writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        val_loss, val_acc, _, _, val_s_recall = evaluate(model, val_loader, device, class_weights=class_weights, print_report=True, info="test")
        dev_loss, dev_acc, _, all_preds, dev_s_recall = evaluate_dev(model, dev_loader, device, class_weights=class_weights, print_report=True, info="Dev", save_mistakes_path=cfg["logging"]["save_mistakes_path"], epoch=epoch+1)

        # 预测分布监控
        counts = np.bincount(all_preds, minlength=cfg["model"]["num_labels"])
        probs = counts / counts.sum()
        print(f"[Monitor] Dev prediction distribution: {probs}")

        # 检查坍缩
        if probs.max() > collapse_threshold:
            print(f"[Warning] Collapse detected at epoch {epoch+1}!"
                  f"One class 占比 {probs.max():.2f} > {collapse_threshold}. 回溯至checkpoint")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                print(f"[Recovery] 模型已回滚到最佳 dev checkpoint (acc={best_acc:.4f}")
            else:
                print("[Recovery] 没有找到已保存的最佳模型，跳过")

        #writer.add_scalar("val/loss", val_loss, epoch)
        #writer.add_scalar("val/acc", val_acc, epoch)

        print(f"Epoch {epoch+1} loss: {dev_loss:.4f}, acc: {dev_acc:.4f}")

        # 保留checkpoint
        if dev_s_recall > best_s_recall:
            best_s_recall = dev_s_recall
            torch.save(model.state_dict(), best_model_path)
            print(f"[Epoch] {epoch+1} New best acc = {dev_acc:.4f}, model saved.")

    print("Training done")
    # writer.close()


if __name__ == "__main__":
    args = parse_args()
    configs = load_config(args.config, args.select)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    for cfg in configs:
        update_config_with_args(cfg, args)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_root, cfg["model"]["model_path"]))
        labeled_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["labeled_path"]))
        dev_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["dev_path"]))
        train_supervised(labeled_ds, dev_ds, tokenizer, cfg, project_root)