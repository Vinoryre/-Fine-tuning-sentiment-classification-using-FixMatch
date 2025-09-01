# auto_hparam_search
import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count
import json
import os
import random
import argparse
import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import yaml
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional, Union


# =======================
# 1) 抽象 Trainer 接口
# =======================
class BaseModelTrainer(ABC):
    """
    继承该类实现训练，预测方法
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs  # 允许透传构造参数

    @abstractmethod
    def train(self, params: Dict[str, Any], data: Dict[str, Any]) -> Any:
        """训练模型，返回模型对象"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Any, data: Dict[str, Any]) -> Dict[str, float]:
        """评估模型，返回指标字典"""
        raise NotImplementedError


# =====================
# 2) 工具函数
# =====================
def dynamic_import(class_path: str):
    """
    动态导入类
    :param class_path:
    :return:
    """
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls


def ensure_dir_for(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_results(df: pd.DataFrame, path: str):
    ensure_dir_for(path)
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.lower().endswith(".json"):
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError("save_path must and with .csv or .json")


def save_best(best_params: Dict[str, Any], best_score: float, path: str):
    ensure_dir_for(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"best_params": best_params, "best_score": best_score},
            f,
            ensure_ascii=False,
            indent=2,
        )


def is_path_like_key(key: str) -> bool:
    """
    判断一个键是否像路径键，用于自动绝对化
    凡是以这些后缀命名的键都按路径处理
    :param key:
    :return:
    """
    if not isinstance(key, str):
        return False
    suffixes = ("path", "dir", "file", "checkpoint", "output")
    return any(key.lower().endswith(suf) for suf in suffixes)


def absolutize_paths(obj: Union[Dict, List, str], root: str) -> Union[Dict, List, str]:
    """
    递归地将对象中所有像路径的键的值由相对路径转化为绝对路径
    仅对键名符合is_path_like_key的项进行处理
    列表会被逐项处理
    只转换字符串值
    :param obj:
    :param root:
    :return:
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                new_obj[k] = absolutize_paths(v, root)
            else:
                if isinstance(v, str) and is_path_like_key(k):
                    new_obj[k] = v if os.path.isabs(v) else os.path.join(root, v)
                else:
                    new_obj[k] = v
        return new_obj
    elif isinstance(obj, list):
        return [absolutize_paths(x, root) for x in obj]
    else:
        return obj


# ====================
# 3) 搜索器
# ====================
@dataclass
class SearchConfig:
    strategy: str       # grid | random
    n_iter: int         # for random
    maximize: str       # metric key to optimize
    mode: str           # max | min
    parallel: bool
    seed: int
    csv_path: str
    best_json_path: str


def _single_run_worker(args) -> Dict[str, Any]:
    """
    为 multiprocessing 准备的顶层函数，避免lambda/闭包不可序列化问题
    :param args:
    :return:
    """
    (
        params,
        trainer_class_path,
        trainer_init_kwargs,
        data,
        maximize_key,
        mode,
    ) = args

    TrainerCls = dynamic_import(trainer_class_path)
    trainer: BaseModelTrainer = TrainerCls(**trainer_init_kwargs)
    model = trainer.train(params, data)
    metrics = trainer.evaluate(model, data)

    if maximize_key not in metrics:
        raise KeyError(f"Metric '{maximize_key}' not found in evaluate() return: {metrics}")

    score = metrics[maximize_key]
    # 统一把score转为“越大越好”(min模型下取负数)
    adj_score = score if mode == "max" else -score

    return {"params": params, **metrics, "_score": adj_score, "_raw_score": score}


class HyperParamSearcher:
    def __init__(self, trainer_class_path: str, trainer_init_kwargs: Dict[str, Any], search_cfg: SearchConfig):
        self.trainer_class_path = trainer_class_path
        self.trainer_init_kwargs = trainer_init_kwargs
        self.search_cfg = search_cfg
        random.seed(self.search_cfg.seed)

    def _enumerate_grid(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _sample_random(self, param_grid: Dict[str, List[Any]], n_iter: int) -> List[Dict[str, Any]]:
        keys = list(param_grid.keys())
        samples = []
        for _ in range(n_iter):
            params = {k: random.choice(param_grid[k]) for k in keys}
            samples.append(params)
        return samples

    def run(self, param_grid: Dict[str, List[Any]], data: Dict[str, Any]) -> Tuple[Dict[str, Any], float, pd.DataFrame]:
        # 生成参数列表
        if self.search_cfg.strategy == "grid":
            trials = self._enumerate_grid(param_grid)
        elif self.search_cfg.strategy == "random":
            trials = self._sample_random(param_grid, self.search_cfg.n_iter)
        else:
            raise ValueError("search.strategy must be 'grid' or 'random'")

        # 准备worker参数
        worker_args = [
            (
                params,
                self.trainer_class_path,
                self.trainer_init_kwargs,
                data,
                self.search_cfg.maximize,
                self.search_cfg.mode
             )
            for params in trials
        ]

        # 并行 or 串行
        if self.search_cfg.parallel:
            n_workers = min(len(worker_args), cpu_count())
            with Pool(processes=n_workers) as pool:
                results = pool.map(_single_run_worker, worker_args)
        else:
            results = [_single_run_worker(a) for a in worker_args]

        # 汇总结果
        df = pd.DataFrame(results)
        # 按 _score(统一越大越好)排序
        df = df.sort_values(by="_score", ascending=False, kind="mergesort").reset_index(drop=True)

        # 取最优
        best_row = df.iloc[0].to_dict()
        best_params = best_row["params"]
        best_raw_score = best_row["_raw_score"]

        # 清理中间列
        df = df.drop(columns=["_score"], errors="ignore")

        # 保存结果
        if self.search_cfg.csv_path:
            save_results(df, self.search_cfg.csv_path)
        if self.search_cfg.best_json_path:
            save_best(best_params, best_raw_score, self.search_cfg.best_json_path)

        print(f"[Best] {self.search_cfg.maximize} ({self.search_cfg.mode}): {best_raw_score:.6f}")
        print(f"[Best Params] {json.dumps(best_params, ensure_ascii=False)}")
        print(f"[Saved] results -> {self.search_cfg.csv_path}, best -> {self.search_cfg.best_json_path}")

        return best_params, best_raw_score, df


# ====================
# 4) CLI & main
# ====================
def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 获取项目根路径
    current_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_root))

    # 转换数据路径为绝对路径
    for key, val in config["data"].items():
        config["data"][key] = os.path.join(project_root, val)

    # 模型路径
    config["trainer"]["init_kwargs"]["model_path"] = os.path.join(
        project_root, config["trainer"]["init_kwargs"]["model_path"]
    )

    # 搜索指标输出目录
    config["search"]["csv_path"] = os.path.join(
        project_root, config["search"]["csv_path"]
    )

    config["search"]["best_json_path"] = os.path.join(
        project_root, config["search"]["best_json_path"]
    )

    # 模型保存目录
    config["data"]["model_output_dir"] = os.path.join(
        project_root, config["data"]["model_output_dir"]
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="Universal Hyper-Parameter Search (Grid/Random) with YAML & OOP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 读取trainer
    trainer_cfg = cfg.get("trainer")
    trainer_class_path = trainer_cfg.get("class_path")
    trainer_init_kwargs = trainer_cfg.get("init_kwargs")

    # 读取search 配置
    s = cfg.get("search")
    search_cfg = SearchConfig(
        strategy=s.get("strategy"),
        n_iter=int(s.get("n_iter")),
        maximize=s.get("maximize"),
        mode=s.get("mode"),
        parallel=bool(s.get("parallel")),
        seed=int(s.get("seed")),
        csv_path=s.get("csv_path"),
        best_json_path=s.get("best_json_path"),
    )

    # 参数空间 & 数据
    param_grid = cfg.get("param_grid")
    data = cfg.get("data")

    # 运行搜索
    searcher = HyperParamSearcher(trainer_class_path, trainer_init_kwargs, search_cfg)
    searcher.run(param_grid, data)


if __name__ == "__main__":
    main()