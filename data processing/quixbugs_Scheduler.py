import os
import json
import time
import logging
from src.pipeline import Pipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# 参数配置
model_name = "gpt-4o"
patch_attempts = 2
level = 1

# 输入和输出路径
json_path = r"D:\UniDebugger-main\dataset\QuixBugs\QuixBugs-master\quixbugs_data.json"
patch_output_dir = r"D:\UniDebugger-main\data processing\quixbugs_patches"
result_output_path = r"D:\UniDebugger-main\data processing\quixbugs_results.json"

os.makedirs(patch_output_dir, exist_ok=True)

# 初始化修复管道
pipe = Pipeline(model_name=model_name,
                # container_id="local_test",
                container_id="defects4j_test_container",
                data_name="quixbugs",
                refinement=False,
                level=level)

# 读取数据
with open(json_path, "r", encoding="utf-8") as f:
    quixbugs_data = json.load(f)

results = {}
success_count = 0

# 遍历样本
for sample in quixbugs_data:
    sample_name = sample["sample_name"]
    buggy_path = sample["buggy_file_path"]
    entry_func = sample["entry_function"]

    info = {
        "project_meta": {
            "bug_name": sample_name,
            "buggy_file_path": buggy_path,
            "project_src_path": os.path.dirname(buggy_path),
            "project_name": "QuixBugs",
            "buggy_number": 0,
            "checkout_dir": "/defects4j"
        },
        "entry_func": entry_func,
        "buggy_code": open(buggy_path, encoding="utf-8").read(),
        "failing_test_cases": "No specific test cases available for QuixBugs"
    }

    print(f"\n=== Running FixAgent on sample: {sample_name} ===")
    try:
        success, patch = pipe.level_1_repair(info, re_patch_num=patch_attempts)

        # 保存补丁到文件夹
        patch_file_path = os.path.join(patch_output_dir, f"{sample_name}_patch.txt")
        with open(patch_file_path, "w", encoding="utf-8") as pf:
            pf.write(patch)

        results[sample_name] = {
            "success": success,
            "patch_file": patch_file_path,
            "error": None
        }

        if success:
            success_count += 1
            print(f"[✓] Plausible patch found for {sample_name}")
        else:
            print(f"[✗] No plausible patch for {sample_name}")

    except Exception as e:
        print(f"[!] Error processing {sample_name}: {e}")
        results[sample_name] = {
            "success": False,
            "patch_file": None,
            "error": str(e)
        }

    # ✅ 每处理一个样本就写入 JSON，防止崩溃丢数据
    results["summary"] = {
        "total": len(quixbugs_data),
        "successful": success_count,
        "success_rate": round(success_count / len(quixbugs_data), 3)
    }
    with open(result_output_path, "w", encoding="utf-8") as rf:
        json.dump(results, rf, indent=2, ensure_ascii=False)

    # 避免触发 429 频率限制
    time.sleep(20)

print("\n✅ All done. Summary:")
print(json.dumps(results["summary"], indent=2, ensure_ascii=False))
