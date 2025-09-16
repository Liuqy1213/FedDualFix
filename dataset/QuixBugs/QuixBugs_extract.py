import os
import json
import logging
import re

# 设置日志格式
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 路径配置（确保路径存在）
base_path = r"D:\UniDebugger-main\dataset\QuixBugs\QuixBugs-master"
buggy_dir = os.path.join(base_path, "java_programs")
fixed_dir = os.path.join(base_path, "correct_java_programs")

# 验证目录
if not os.path.isdir(buggy_dir) or not os.path.isdir(fixed_dir):
    logging.error("Buggy 或 Fixed 目录不存在，请检查路径！")
    exit(1)

# 获取文件名映射（不带扩展名）
buggy_files = {os.path.splitext(f)[0]: os.path.join(buggy_dir, f)
               for f in os.listdir(buggy_dir) if f.endswith(".java")}
fixed_files = {os.path.splitext(f)[0]: os.path.join(fixed_dir, f)
               for f in os.listdir(fixed_dir) if f.endswith(".java")}

records = []
for sample_name, buggy_path in buggy_files.items():
    if sample_name not in fixed_files:
        logging.warning(f"缺少修复文件：{sample_name}")
        continue

    fixed_path = fixed_files[sample_name]

    # 尝试读取代码以确定 entry function
    entry_function = "main"  # 默认设为 main
    try:
        with open(buggy_path, 'r', encoding='utf-8') as f:
            code = f.read()
    except Exception as e:
        logging.error(f"读取失败：{buggy_path}，错误：{e}")
        continue

    if "public static void main" not in code:
        # 查找其他 public static 函数
        match = re.search(r'public static [\w<>\[\]]+\s+(\w+)\s*\(', code)
        if match:
            entry_function = match.group(1)
        else:
            # 最后兜底用类名的小写
            entry_function = sample_name.lower()

    # 构建记录
    record = {
        "sample_name": sample_name,
        "buggy_file_path": buggy_path,
        "fixed_file_path": fixed_path,
        "entry_function": entry_function,
        "metadata": {}
    }
    records.append(record)
    logging.info(f"已处理：{sample_name} (入口函数：{entry_function})")

# 保存为 JSON 文件
output_path = os.path.join(base_path, "quixbugs_data.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, indent=4, ensure_ascii=False)

logging.info(f"处理完成，共生成 {len(records)} 条记录，保存至：{output_path}")
