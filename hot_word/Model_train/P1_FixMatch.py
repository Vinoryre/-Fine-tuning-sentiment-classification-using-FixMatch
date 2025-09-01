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


def strong_augment(text):
    # 强增强: 随机mask 15%字符
    words = list(text)
    n_mask = max(1, int(len(words) * 0.15))
    for _ in range(n_mask):
        idx = random.randint(0, len(words)-1)
        words[idx] = "[MASK]"
    return "".join(words)


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
                            text = text.strip()
                            if not text:
                                print(f"Warning: line {i+2} if empty text, skipped")
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
                            text = parts[0].strip()  # 去掉首尾空格
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
    strong_texts = [strong_augment(t) for t in batch]
    enc_w = tokenizer(weak_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    enc_s = tokenizer(strong_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc_w, enc_s, original_texts


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
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = output.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits


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
def evaluate_dev(model, loader, device, class_weights=None, print_report=False, info=None, save_mistakes_path=None, epoch=None, save_every=5, pos_threshold=0.6, neg_threshold=0.6):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    mistake_smaples = []  # 存放错误样本

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


# === EMA更新 ===
@torch.no_grad()
def ema_update(ema_model, model, decay):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1-decay)


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

    # writer = SummaryWriter(log_dir=log_dir)

    label_names = ["正向", "负向", "中性"]
    labeled_size = len(train_ds)
    full_unlabeled_size = len(unlabeled_ds)

    print(labeled_size)
    print(full_unlabeled_size)

    # 梯度积累模拟大BATCH
    physical_B_U = 64  # 实际送GPU的无标签batch大小
    accum_steps_u = cfg["training"]["batch_u"] // physical_B_U
    # accum_counter = 0

    best_normal_acc = 0.0
    best_ema_acc = 0.0
    best_model_dir = os.path.join(project_root, cfg["model"]["best_model_dir"])
    os.makedirs(best_model_dir, exist_ok=True)
    global_step = 0
    collapse_threshold = cfg["training"].get("collage_threshold", 0.95)

    loader_l = DataLoader(train_ds, batch_size=cfg["training"]["batch_l"], shuffle=True,
                          collate_fn=lambda b: collate_labeled(b, tokenizer, cfg["dataset"]["max_len"]))
    loader_u = DataLoader(unlabeled_ds, batch_size=physical_B_U, shuffle=True,
                          collate_fn=lambda b: collate_unlabeled(b, tokenizer, cfg["dataset"]["max_len"]))
    it_u = iter(loader_u)

    for epoch in range(cfg["training"]["epochs"]):
        k = epoch + 1
        # 计算无标签采样比例
        # alpha = 1
        # ratio = (alpha * k / cfg["training"]["epochs"]) * full_unlabeled_size / labeled_size
        # ratio = min(ratio, full_unlabeled_size / labeled_size)

        print(f"=== Epoch {k}, 使用全量无标签数据 ===")
        # unlabeled_subset = get_unlabeled_subset(unlabeled_ds, ratio, labeled_size)

        for enc_l, y_l in loader_l:
            # 有标签数据移动到GPU
            enc_l, y_l = {k: v.to(device) for k, v in enc_l.items()}, y_l.to(device)

            with autocast(enabled=cfg["training"]["use_amp"]):
                # 有标签损失
                logits_l = model(**enc_l)
                loss_l = F.cross_entropy(logits_l, y_l)

                # 累加无标签损失
                loss_u_total = 0.0
                for _ in range(accum_steps_u):
                    # 取无标签batch
                    try:
                        enc_u_w, enc_u_s, orig_u_texts = next(it_u)
                    except StopIteration:
                        it_u = iter(loader_u)
                        enc_u_w, enc_u_s, orig_u_texts = next(it_u)

                    # 无标签数据移动到GPU
                    enc_u_w = {k: v.to(device) for k, v in enc_u_w.items()}
                    enc_u_s = {k: v.to(device) for k, v in enc_u_s.items()}

                    # 无标签弱增强->伪标签
                    with torch.no_grad():
                        logits_u_w = ema_model(**enc_u_w)
                        probs_u_w = torch.softmax(logits_u_w, dim=-1)
                        max_probs, pseudo_labels = torch.max(probs_u_w, dim=-1)
                        # 打印调试阈值
                        pmax = probs_u_w.max(dim=1).values

                        IGNORE_INDEX = -1
                        num_labeled_in_batch = int((y_l.long() != IGNORE_INDEX).sum())

                        printed = {
                            "pmax_mean": pmax.mean().item(),
                            "pmax_90": pmax.kthvalue(int(0.9 * len(pmax))).values.item(),
                            "selected_frac": (pmax > cfg["training"]["threshold"]).float().mean().item(),
                            "lr": optimizer.param_groups[0]['lr'],
                            "num_labeled_in_batch": num_labeled_in_batch
                        }
                        if global_step % 1000 == 0:
                            print(printed)

                        # 强增强一致性损失
                        logits_u_s = model(**enc_u_s)
                        loss_u = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none')
                        # loss_u 中有低置信度的样本，要丢弃
                        mask = (max_probs >= cfg["training"]["threshold"]).float()
                        loss_u = (loss_u * mask).mean()
                        loss_u_total += loss_u  # 直接进行累加，不除以7

            # 总损失
            loss = loss_l + cfg["training"]["lambda_u"] * loss_u_total

            # backward
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()
            ema_update(ema_model, model, cfg["training"]["ema_decay"])

            # tensorboard
            # writer.add_scalar("loss/labeled", loss_l.item(), global_step)
            # # #writer.add_scalar("loss/unlabeled", loss_u.item(), global_step)
            # # writer.add_scalar("loss/total", loss.item(), global_step)
            # # #writer.add_scalar("stats/selected_fraction", mask.mean().item(), global_step)

            # 文本写入tensorboard: 有标签和无标签
            # if global_step % cfg["training"]["sample_log_freq"] == 0:
            #     example_ids = enc_l['input_ids'][0]
            #     example_text = tokenizer.decode(example_ids, skip_special_tokens=True)
            #     true_label_name = label_names[y_l[0].item()]
            #     writer.add_text("sample/labeled", f"text: {example_text}\nlabels: {true_label_name}", global_step)
            #
            #     unlabeled_text = orig_u_texts[0]
            #     pseudo_label_idx = pseudo_labels[0].item()
            #     pseudo_label_name = label_names[pseudo_label_idx] if pseudo_label_idx < len(label_names) else str(pseudo_label_idx)
            #     writer.add_text("sample/unlabeled", f"text: {unlabeled_text}\npseudo_label: {pseudo_label_name}", global_step)

            global_step += 1

        # === 每个epoch验证 ===
        # val_loss_test, normal_accuracy_test, _, _, val_s_recall_test = evaluate(model, test_loader, device, class_weights=None, print_report=True, info="val_test")
        ema_loss_test, ema_accuracy_test, _, _, ema_s_recall_test = evaluate(ema_model, test_loader, device, class_weights=None, print_report=True, info="ema_test")
        # val_loss_dev, normal_accuracy_dev, _, _, val_s_recall_dev = evaluate_dev(model, dev_loader, device, class_weights=None, print_report=True, info="val_dev", save_mistakes_path=cfg["logging"]["save_mistakes_path"], epoch=epoch+1)
        ema_loss_dev, ema_accuracy_dev, _, all_preds, ema_s_recall_dev = evaluate_dev(ema_model, dev_loader, device, class_weights=None, print_report=True, info="ema_dev", save_mistakes_path=cfg["logging"]["save_mistakes_path"], epoch=epoch+1)
        # writer.add_scalar("test/loss_normal", val_loss, epoch)
        # writer.add_scalar("test/accuracy_normal", normal_accuracy, epoch)
        # writer.add_scalar("test/loss_ema", ema_loss, epoch)
        # writer.add_scalar("test/accuracy_ema", ema_accuracy, epoch)

        # 预测分布监控
        counts = np.bincount(all_preds, minlength=cfg["model"]["num_labels"])
        probs = counts / counts.sum()
        print(f"[Monitor] ema Dev prediction distribution: {probs}")

        best_model_path = os.path.join(project_root, best_model_dir, "fixmatch_model.pt")

        # 检查坍缩
        if probs.max() > collapse_threshold:
            print(f"[Warning] Collapse detected at epoch {epoch+1} !"
                  f"One class 占比 {probs.max():.2f} > {collapse_threshold}. 回溯至checkpoint")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                print(f"[Recovery] 模型已回滚到最佳 dev checkpoint (acc={ema_accuracy_dev:.4f}")
            else:
                print("[Recovery] 没有找到已保存的最佳模型，跳过")

        print(f"Epoch {epoch+1} loss: {ema_loss_dev:.4f}, acc: {ema_accuracy_dev:.4f}")

        # 仅在性能更好的时候才保存checkpoint(只使用ema模型)
        if ema_accuracy_test > best_ema_acc:
            best_ema_acc = ema_accuracy_test
            torch.save(ema_model.state_dict(), best_model_path)
            print(f"Saved new best EMA model at epoch {k} (acc={best_ema_acc:.4f})")

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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_root, cfg["model"]["model_path"]))
        labeled_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["labeled_path"]))
        unlabeled_ds = UnlabeledDataset(os.path.join(project_root, cfg["dataset"]["unlabeled_path"]))
        dev_ds = LabeledDataset(os.path.join(project_root, cfg["dataset"]["dev_path"]))

        print(f"===== Running config: {cfg.get('name', 'default')} =====")
        train(labeled_ds, unlabeled_ds, dev_ds, tokenizer, cfg, project_root)

    print("end")
