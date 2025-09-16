# src/patch.py
import os
import re
import logging
import signal
from typing import List, Optional

from .parse import (
    NoCodeError,
    matching_lines,
    matching_with_comments,
    unique_matching,
    search_valid_line,
)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("TimeOut")

class NotPatchError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
        print(message)

def _format_patch_lines(patch: str) -> List[str]:
    # 把 @@ hunk 头与紧随其后的首行代码拆行，便于逐行处理
    return re.compile(r'(@@[\s\d\+\-\,]+@@)(\s+[^\n]+)').sub(r'\1\n\2', patch).splitlines()

def _format_code(code_lines: List[str]) -> List[str]:
    """
    对原代码做一些轻度的行级规整（拼接被错误换行的修饰符等）。
    依据原 FixAgent 的启发式，维持兼容。
    """
    out = list(code_lines)
    for i, line in enumerate(list(out)):
        concat = False
        for k in ["public", "private", "protected"]:
            if line.replace(" ", "") == k:
                concat = True
                out[i] = ""
                if i + 1 < len(out):
                    out[i + 1] = k + " " + out[i + 1].lstrip()
                break

        if (not concat and i < len(out) - 1 and "class" not in line
                and not line.strip().startswith("*")
                and ("//" not in line and "/*" not in line)
                and re.search(r'[A-Za-z0-9]$', line or "")):
            # 把“被错误换行”的两行拼回去
            out[i] = (line or "").rstrip() + " " + (out[i + 1] or "").lstrip()
            out[i + 1] = ""
    return out

def _is_a_patch(patch_lines: List[str]) -> bool:
    has_hunk = any(re.match(r'^@@\s-\d+,\d+\s\+\d+,\d+\s@@.*', (ln or '').strip()) for ln in patch_lines)
    has_sign = any(re.match(r'^[-+](\s|\t)+.*$', (ln or '').strip()) for ln in patch_lines)
    return bool(has_hunk and has_sign)

def _find_a_matched_line(
    pidx: int,
    pline: str,
    code_lines: List[str],
    patch_lines: List[str],
    lag: int = -1,
    existing: bool = False
) -> int:
    matched = matching_lines(pline, code_lines)
    if lag >= 0:
        matched = [m for m in matched if m > lag]
    if len(matched) == 1:
        return matched[0]

    perfect = matching_with_comments(pline, matched, code_lines)
    if len(perfect) == 1:
        return perfect[0]

    return unique_matching(patch_lines, code_lines, pidx, resp_cur_line=pline, existing=existing)

def patching(patch: str, raw_code_lines: List[str]) -> str:
    """
    把 unified-diff 风格的补丁应用到原代码（行级，非 git apply）。
    返回“打好补丁”的代码文本。
    """
    patch_lines = _format_patch_lines(patch)
    if not _is_a_patch(patch_lines):
        raise NoCodeError(f"Not a patch!\n{patch}")

    assert isinstance(raw_code_lines, list)
    code_lines = _format_code(raw_code_lines)

    # 目标：在 code_lines 上，按 -/+ 序列执行删除/替换/插入
    patched = list(code_lines)
    unpatched = []
    to_patch = 0
    replace_idx, prev_patch_idx = -1, -1  # 记录上一处变更位置与行

    for pidx, pline in enumerate(patch_lines):
        if re.search(r'^[-](\s|\t)+.*$', pline or ""):  # 删除
            to_patch += 1
            if prev_patch_idx >= 0 and prev_patch_idx + 1 == pidx and patch_lines[prev_patch_idx].startswith('-'):
                # 连续多行删除
                if replace_idx + 1 < len(patched):
                    patched[replace_idx + 1] = ""
                replace_idx, prev_patch_idx = replace_idx + 1, pidx
            else:
                match_idx = _find_a_matched_line(
                    pidx, (pline or "")[1:].lstrip(), code_lines, patch_lines,
                    lag=replace_idx, existing=True
                )
                if match_idx >= 0:
                    patched[match_idx] = ""
                    replace_idx, prev_patch_idx = match_idx, pidx
                else:
                    logging.warning(f"Cannot patch {pline}!")
                    unpatched.append((pidx, pline))

        elif re.search(r'^[+](\s|\t)+.*$', pline or ""):  # 新增
            to_patch += 1
            if prev_patch_idx >= 0 and prev_patch_idx + 1 == pidx:
                # 与上一行（可能是 - 或 +）形成“替换或多行追加”
                if patch_lines[prev_patch_idx].startswith('-'):
                    patched[replace_idx] = (pline or "")[1:].rstrip()
                elif patch_lines[prev_patch_idx].startswith('+'):
                    if isinstance(patched[replace_idx], str):
                        patched[replace_idx] += ("\n" + (pline or "")[1:].rstrip())
                    elif isinstance(patched[replace_idx], list):
                        patched[replace_idx] = patched[replace_idx][:-1] + [(pline or "")[1:].rstrip(), patched[replace_idx][-1]]
                prev_patch_idx = pidx
            else:
                # 尝试“邻居定位”后再插入
                pre_valid = search_valid_line(patch_lines, pidx, "pre")
                if pre_valid is not None:
                    unique_idx = _find_a_matched_line(pre_valid[0], pre_valid[1], code_lines, patch_lines, lag=replace_idx)
                    if unique_idx >= 0:
                        patched[unique_idx] += ("\n" + (pline or "")[1:].rstrip())
                        replace_idx, prev_patch_idx = unique_idx, pidx
                        continue
                post_valid = search_valid_line(patch_lines, pidx, "post", existing=code_lines)
                if post_valid is not None:
                    unique_idx = _find_a_matched_line(post_valid[0], post_valid[1], code_lines, patch_lines, lag=replace_idx)
                    if unique_idx >= 0:
                        patched[unique_idx] = [(pline or "")[1:].rstrip(), patched[unique_idx]]
                        replace_idx, prev_patch_idx = unique_idx, pidx
                        continue

                logging.warning(f"Cannot patch! {pline}")
                unpatched.append((pidx, pline))

    if len(unpatched) == to_patch:
        # 一行都没打上
        raise NotPatchError("No lines applied from the patch.")

    if unpatched:
        for (pidx, pline) in unpatched:
            logging.debug(f"Unpatched lines: #{pidx}\t{pline}")

    # 拼回文本
    res = []
    for p in patched:
        if not p:
            continue
        if isinstance(p, str):
            res.append(p)
        elif isinstance(p, list):
            res.extend(p)
        else:
            raise TypeError(f"Unsupported patched element type: {type(p)}")
    return "\n".join(res).strip()

def testing(root_test_dir: str, container) -> int:
    """
    在 defects4j 容器里编译并运行测试，返回 failing 测例数。
    """
    env_vars = {'JAVA_TOOL_OPTIONS': '-Dfile.encoding=UTF8'}
    logging.info("# Compiling...")
    compile_result = container.exec_run(
        "sh -c 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64 && defects4j compile'",
        workdir=root_test_dir, environment=env_vars
    ).output.decode('utf-8', errors='ignore')

    if "BUILD FAILED" in compile_result:
        logging.warning(f"Compile Failed\n{compile_result}")
        return -1

    logging.info("# Testing...")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5 * 60)
    try:
        test_result = container.exec_run(
            "sh -c 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-arm64 && defects4j test'",
            workdir=root_test_dir, stderr=True, stdout=True
        ).output.decode('utf-8', errors='ignore')
        signal.alarm(0)
    except TimeoutException:
        logging.warning("Timeout!")
        return -1
    except Exception as e:
        logging.error(f"Errors during testing: {e}")
        return -1

    m = re.search(r'Failing tests:\s*(\d+)', test_result)
    if m:
        # 临时文件清理交给调用方自行处理；这里不删除本地文件
        return int(m.group(1))
    else:
        logging.error(f"Test results parse error\n{test_result}")
        return -1

def patching_and_testing(patch: str, project_meta: dict, container_id: Optional[str] = None) -> Optional[bool]:
    """
    把补丁应用到目标文件并在容器里跑测试。
    - 返回 True  : 全部测试通过
    - 返回 False : 仍有失败
    - 返回 None : 补丁格式无效/无法应用
    - container_id 为 None 时，走 dry-run，直接返回 True（按“可行”占位）
    """
    if container_id is None:
        logging.info("[Dry Run] Skip container-based testing, return plausible=True.")
        return True

    try:
        import docker
    except Exception:
        logging.warning("docker SDK unavailable; skip testing.")
        return None

    client = docker.from_env()
    container = client.containers.get(container_id)

    main_dir = "/defects4j"
    bug_name = f"{project_meta['project_name']}_{project_meta['buggy_number']}"
    project_dir = os.path.join(project_meta['checkout_dir'], f"{bug_name}_buggy")

    logging.info("# Checking out...")
    container.exec_run(f"rm -rf {project_dir}", workdir=main_dir)
    container.exec_run(
        f"defects4j checkout -p {project_meta['project_name']} -v {project_meta['buggy_number']}b -w {project_dir}",
        workdir=main_dir
    )

    # 读取原代码
    code_bytes = container.exec_run(f"cat {project_meta['buggy_file_path']}", workdir=main_dir).output
    buggy_code = code_bytes.decode('utf-8', errors='ignore')

    # 应用补丁
    try:
        patched_code = patching(patch, buggy_code.splitlines())
    except (NoCodeError, NotPatchError) as e:
        logging.warning(f"Cannot apply patch: {e}")
        return None

    # 写回容器
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".java", encoding="utf-8") as tf:
        tf.write(patched_code)
        tmp_path = tf.name
    try:
        subprocess.run(
            ["docker", "cp", tmp_path, f"{container_id}:{os.path.join(main_dir, project_meta['buggy_file_path'])}"],
            check=True
        )
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 跑测试
    return testing(os.path.join(main_dir, project_dir), container) == 0
