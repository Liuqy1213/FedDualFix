import os
import json
import logging
import hashlib
import yaml
from typing import Any, Dict, List, Tuple, Optional


# ---------------- I/O helpers ----------------

def _ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of a file path exists."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def temporary_save(fixer_response: Dict[str, str], project_meta: Dict[str, Any], base_dir: str = "temp") -> None:
    """
    将 Fix 阶段的响应临时落盘，默认保存到 temp/fixer/<proj>_<id>.patch|txt
    """
    proj = project_meta.get("project_name", "proj")
    num = project_meta.get("buggy_number", "X")
    patch_path = os.path.join(base_dir, "fixer", f"{proj}_{num}.patch")
    ori_path   = os.path.join(base_dir, "fixer", f"{proj}_{num}.txt")

    _ensure_parent_dir(patch_path)
    _ensure_parent_dir(ori_path)

    # 允许缺项（稳一手）
    patch_text = fixer_response.get("aim", "")
    ori_text   = fixer_response.get("ori", "")

    with open(patch_path, "w", encoding="utf-8", newline="") as wf:
        wf.write(patch_text)
    with open(ori_path, "w", encoding="utf-8", newline="") as wf:
        wf.write(ori_text)


def read_yaml(file_path: str) -> Any:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # may return dict/list/None


def get_content(file_path: str) -> Dict[str, str]:
    """
    读取文本文件内容，统一返回字典：
    {"content": "...", "err": ""} 或 {"content": "", "err": "xxx"}
    """
    if not os.path.exists(file_path):
        return {"content": "", "err": f"{file_path} does not exist"}
    if not os.path.isfile(file_path):
        return {"content": "", "err": f"{file_path} is not a file"}
    try:
        with open(file_path, "r", encoding="utf-8") as rf:
            return {"content": rf.read(), "err": ""}
    except Exception as e:
        return {"content": "", "err": f"read error: {e}"}


def read_json(filepath: str) -> Any:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not exists: {filepath}")
    if not filepath.endswith(".json"):
        raise ValueError(f"Not a .json file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def json_pretty_dump(obj: Any, filename: str) -> None:
    _ensure_parent_dir(filename)
    with open(filename, "w", encoding="utf-8", newline="") as fw:
        json.dump(
            obj,
            fw,
            sort_keys=True,
            indent=4,
            separators=(",", ": "),
            ensure_ascii=False,
        )


# ---------------- logging & experiment ----------------

def logging_activate(record_dir: str) -> None:
    os.makedirs(record_dir, exist_ok=True)
    log_file = os.path.join(record_dir, "running.log")

    # 清理现有 root handlers，避免重复日志
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def dump_exp(result_save_dir: str, params: Dict[str, Any]) -> Tuple[str, str]:
    """
    基于参数生成 hash_id，创建实验目录，并写入 params.json，返回 (record_dir, hash_id)
    """
    os.makedirs(result_save_dir, exist_ok=True)
    # 排序后转字符串，保证哈希稳定
    serial = str(sorted((k, v) for k, v in params.items()))
    hash_id = hashlib.md5(serial.encode("utf-8")).hexdigest()[:8]
    record_dir = os.path.join(result_save_dir, hash_id)
    os.makedirs(record_dir, exist_ok=True)

    json_pretty_dump(params, os.path.join(record_dir, "params.json"))
    logging_activate(record_dir)
    return record_dir, hash_id


# ---------------- small file utils ----------------

def return_lines(file_path: str) -> List[str]:
    if os.path.exists(file_path) and os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as rf:
            return [ln.rstrip("\n") for ln in rf.readlines()]
    return []


def write_line(file_path: str, line: str) -> None:
    _ensure_parent_dir(file_path)
    with open(file_path, "a", encoding="utf-8", newline="") as wf:
        wf.write(line + "\n")

