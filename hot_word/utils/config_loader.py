import yaml


def load_config(config_path, select=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config_all = yaml.safe_load(f)

    # 多配置模型
    if isinstance(config_all, dict) and "configs" in config_all:
        if select is None:
            return list(config_all["configs"].values())
        else:
            return [config_all["configs"][select]]
    else:
        return [config_all]
