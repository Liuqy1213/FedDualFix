import os
import logging
from typing import Optional, Dict, Any
from retry import retry

from ..utils import read_yaml
from ..parse import (
    parse_code, parse_exp, is_valid_line,
    unique_matching, search_valid_line
)
from .agent import Agent, RetryError, NoCodeError
from ..prompts.tokens import calculate_token, token_limit


class SliceAgent(Agent):
    """
    Extract a suspicious code segment around the bug.
    Returns both the raw segment text and basic metadata (start/end indices).
    """

    def parse_response(self, response: str, raw_code: str):
        # 1) 收集 agent 返回的“候选片段”
        parts = [p.strip() for p in parse_code(response) if p.strip()]
        if not parts:
            raise NoCodeError("No code block found in model response.")
        segment = "\n".join(parts)
        if "===" in segment:
            segment = segment[: segment.find("===")].strip()

        seg_lines = segment.splitlines()
        raw_lines = raw_code.splitlines()
        seg_s, seg_e = -1, -1  # raw_code 中的起止行号（0-based，闭区间）

        # 2) 起点：seg_lines 中从前往后选首个有效行，去 raw 中唯一匹配
        for cur, line in enumerate(seg_lines):
            if is_valid_line(line):
                unique_idx = unique_matching(seg_lines, raw_lines, cur)
                if unique_idx >= 0:
                    seg_s = unique_idx
                    break
        if seg_s == -1:
            logging.warning("Cannot locate beginning line of the segment!")

        # 3) 终点：从后往前找 seg_lines 的最后一个有效行，去 raw 中唯一匹配
        for cur_rev, line in enumerate(reversed(seg_lines)):
            if is_valid_line(line):
                cur = len(seg_lines) - 1 - cur_rev  # 还原成正向索引
                unique_idx = unique_matching(seg_lines, raw_lines, cur)
                if unique_idx >= 0:
                    seg_e = unique_idx
                    break

        # 4) 构造 raw 中的“真实片段”与回退策略
        if seg_s < 0 and seg_e < 0:
            raise RetryError("Cannot locate the suspicious segment!\n" + segment)
        elif seg_s >= 0 and seg_e >= 0 and seg_s <= seg_e:
            real_seg = "\n".join(raw_lines[max(0, seg_s - 10): min(len(raw_lines), seg_e + 10)])
        else:
            # 只匹配到一端时的回退：向两侧扩 50 行（忽略注释块行）
            if seg_e >= 0:
                real_seg_lines = []
                for i in range(min(len(raw_lines) - 1, seg_e + 10), -1, -1):
                    if len(real_seg_lines) >= 50:
                        break
                    if not (raw_lines[i].strip().startswith('*') or raw_lines[i].strip().startswith('/*')):
                        real_seg_lines.append(raw_lines[i])
                real_seg = "\n".join(reversed(real_seg_lines))
                seg_s = max(0, seg_e - len(real_seg_lines) + 1)
            else:
                real_seg_lines = []
                for i in range(max(0, seg_s - 10), min(len(raw_lines), seg_s + 50)):
                    if len(real_seg_lines) >= 50:
                        break
                    if not (raw_lines[i].strip().startswith('*') or raw_lines[i].strip().startswith('/*')):
                        real_seg_lines.append(raw_lines[i])
                real_seg = "\n".join(real_seg_lines)
                seg_e = min(len(raw_lines) - 1, seg_s + len(real_seg_lines) - 1)

        meta = {"start": seg_s, "end": seg_e, "len": seg_e - seg_s + 1 if seg_s >= 0 and seg_e >= 0 else None}
        return {"aim": real_seg, "exp": parse_exp(response), "ori": response, "metrics": meta}

    def _generate_core_msg(self, info: Dict[str, Any], pre_agent_resp: Dict[str, Any]):
        self.core_msg = "The following code contains a bug:\n" + info["buggy_code"]
        self._shared_msg(info, pre_agent_resp)
        if "coverage_report" in info and calculate_token(self.core_msg + info["coverage_report"]) <= token_limit[self.model_name]["overall"]:
            self.core_msg = "Code coverage for failed testcases:\n" + info["coverage_report"] + "\n" + self.core_msg
        logging.info(f"Current core message tokens: {calculate_token(self.core_msg)}")

    @retry((NoCodeError, RetryError), tries=3, delay=5)
    def run(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Optional[Dict[str, Any]] = None,
        *,
        level: Optional[int] = None,
        signals: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        logging.info("## Running SliceAgent...")
        if pre_agent_resp is None:
            pre_agent_resp = {}
        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/slicer.yaml"))
        if self.core_msg is None:
            self._generate_core_msg(info, pre_agent_resp)

        reply = self.send_message([
            {"role": "system", "content": prompts["sys"]},
            {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]}
        ])
        return self.parse_response(reply, raw_code=info["buggy_code"])

    def refine(self, assist_resp: str, *args, **kwargs):
        prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/slicer.yaml")
        refine_prompt = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/refine.yaml"))
        prompts = read_yaml(prompts_path)
        reply = self.send_message([
            {"role": "system",    "content": prompts["sys"]},
            {"role": "user",      "content": self.core_msg + "\n" + prompts["end"]},
            {"role": "assistant", "content": assist_resp},
            {"role": "user",      "content": "\nModifying your isolated code segment cannot fix the bug:\n" + refine_prompt["slicer"]}
        ])
        return self.parse_response(reply, raw_code=kwargs.get("raw_code", ""))






