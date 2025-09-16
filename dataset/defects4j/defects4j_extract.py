import os
import shutil

# 输入仓库路径
input_root = r"D:\UniDebugger-main\dataset\astor\examples"
# 输出提取目录
output_root = r"D:\UniDebugger-main\dataset\defects4j"
os.makedirs(output_root, exist_ok=True)

bug_names = ["chart_1", "lang_1", "math_1"]

def find_all_java_files(base_dir):
    java_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".java"):
                java_files.append(os.path.join(root, f))
    return java_files

for bug in bug_names:
    print(f"\n📦 正在处理：{bug}")
    input_dir = os.path.join(input_root, bug)
    output_dir = os.path.join(output_root, bug)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 查找 buggy.java（优先从 src/ 下提取任意一个 .java 文件）
    src_candidates = [os.path.join(input_dir, d) for d in ["src/main/java", "src/java", "src"]]
    found_buggy = False
    for path in src_candidates:
        if os.path.exists(path):
            java_files = find_all_java_files(path)
            if java_files:
                shutil.copy(java_files[0], os.path.join(output_dir, "buggy.java"))
                print(f"  ✅ buggy.java 提取自：{java_files[0]}")
                found_buggy = True
                break
    if not found_buggy:
        print("  ❌ 未找到任何 buggy.java")

    # 2. patch.diff
    patch_file = os.path.join(input_dir, "patch.diff")
    if os.path.exists(patch_file):
        shutil.copy(patch_file, os.path.join(output_dir, "patch.diff"))
        print(f"  ✅ patch.diff 拷贝成功")
    else:
        print("  ⚠️ patch.diff 缺失")

print("\n✅ 全部提取完成，已保存至：", output_root)
