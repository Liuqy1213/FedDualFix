import os
import logging
from typing import List, Tuple
from retry import retry

from .agent import Agent, RetryError
from ..parse import parse_code, parse_exp
from ..utils import read_yaml
from ..prompts.tokens import calculate_token, token_limit


ALLOW_EXT = {".java", ".py", ".cc", ".cpp", ".c", ".cs", ".go", ".kt", ".scala", ".swift", ".rs", ".js", ".ts"}


def _build_tree(root: str, max_depth: int = 2, max_entries: int = 200) -> str:
    """
    A lightweight, cross-platform 'tree' summary of the source dir.
    """
    root = os.path.abspath(root)
    lines: List[str] = []
    n = 0
    for cur, dirs, files in os.walk(root):
        depth = cur.replace(root, "").count(os.sep)
        if depth > max_depth:
            continue
        indent = "  " * depth
        rel = os.path.relpath(cur, root)
        lines.append(f"{indent}{'.' if rel == '.' else rel}/")
        for f in sorted(files):
            if n >= max_entries:
                lines.append(f"{indent}  ...")
                return "\n".join(lines)
            if os.path.splitext(f)[1].lower() in ALLOW_EXT:
                lines.append(f"{indent}  {f}")
                n += 1
    return "\n".join(lines)


class FocusAgent(Agent):
    """Static dependency/context focus: propose a small set of candidate files."""

    def parse_response(self, response: str, project_src_path: str, *, top_k: int = 5):
        files_block = "\n".join([p.strip() for p in parse_code(response) if p.strip()])
        if "===" in files_block:
            files_block = files_block[: files_block.find("===")].strip()

        cands = [f.strip() for f in files_block.splitlines() if f.strip()]
        valid_files: List[str] = []
        for f in cands:
            full = os.path.join(project_src_path, f)
            if os.path.exists(full) and "Test" not in f and os.path.splitext(f)[1].lower() in ALLOW_EXT:
                valid_files.append(os.path.relpath(full, project_src_path))

        if not valid_files:
            raise RetryError(f"No valid files parsed from:\n{files_block}")

        if len(valid_files) > top_k:
            logging.warning("Too many valid files, truncating to top_k=%d", top_k)
            valid_files = valid_files[:top_k]

        return {"aim": valid_files, "exp": parse_exp(response), "ori": response}

    def _generate_core_msg(self, info: dict):
        src_root = info["project_meta"]["project_src_path"]
        structure = _build_tree(src_root, max_depth=2)

        parts = []
        if info.get("packages"):
            parts.append("Imported packages in the bug-located code file:\n" + info["packages"])
        if info.get("failing_test_cases"):
            parts.append("The code fails on this test:\n" + info["failing_test_cases"])
        parts.append("Structure of source code directory:\n" + structure)

        self.core_msg = "\n".join(parts)

        if "coverage_report" in info and calculate_token(self.core_msg + info["coverage_report"]) <= token_limit[self.model_name]["overall"]:
            self.core_msg = "Code coverage for failed testcases:\n" + info["coverage_report"] + "\n" + self.core_msg

    @retry((RetryError,), tries=3, delay=5)
    def run(self, info: dict, *args, top_k: int = 5, **kwargs):
        logging.info("## Running FocusAgent...")
        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/repofocus.yaml"))
        if self.core_msg is None:
            self._generate_core_msg(info)
            # 复用共享上下文（若上游提供）
            self._shared_msg(info, kwargs.get("pre_agent_resp", {}))

        reply = self.send_message([
            {"role": "system", "content": prompts["sys"]},
            {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]}
        ])
        return self.parse_response(reply, info["project_meta"]["project_src_path"], top_k=top_k)





