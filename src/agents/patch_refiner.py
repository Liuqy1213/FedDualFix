import os
import logging
from typing import Optional, Dict, Any
from retry import retry

from ..utils import read_yaml
from ..parse import parse_code, parse_exp
from ..prompts.tokens import calculate_token, token_limit
from .agent import Agent, NoCodeError


class PatchRefiner(Agent):
    """
    Former 'FixerPro'. Iteratively refine low-confidence patches at L3.
    - Accept last patch + failing diagnostics
    - Accept alignment/confidence signals to steer refinement
    """

    def parse_response(self, response: str):
        parts = parse_code(response)
        if not parts:
            raise NoCodeError("No code block found in model response.")
        patch = parts[0].strip()
        if "===" in patch:
            patch = patch[: patch.find("===")].strip()
        return {"aim": patch, "exp": parse_exp(response), "ori": response}

    def _generate_core_msg(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Dict[str, Any],
        *,
        last_patch: str,
        test_res: str,
        signals: Optional[Dict[str, Any]] = None,
    ):
        # 基础上下文：有定位则带标注，否则用原始代码
        if "locator" in pre_agent_resp:
            self.core_msg = (
                "The following code contains a bug with suspicious lines labeled:\n"
                + pre_agent_resp["locator"]
            )
        else:
            self.core_msg = "The following code contains a bug:\n" + info["buggy_code"]

        # 共享上下文（失败用例、摘要、helper等）
        self._shared_msg(info, pre_agent_resp)

        # 覆盖率（若在 token 预算内）
        if "coverage_report" in info and calculate_token(self.core_msg + info["coverage_report"]) <= token_limit[self.model_name]["overall"]:
            self.core_msg = "Code coverage for failed testcases:\n" + info["coverage_report"] + "\n" + self.core_msg

        # 失败原因与上一版补丁
        tail = []
        if last_patch:
            tail.append("Previous attempt (patch):\n" + last_patch)
        if test_res:
            tail.append("The previous attempt fails because:\n" + test_res)

        # 对齐/置信度等信号
        if signals:
            hints = []
            if "D_aln" in signals:
                hints.append(f"Alignment score (D_aln): {signals['D_aln']:.3f}. Improve consistency with textual intent.")
            if "C_prev" in signals:
                hints.append(f"Previous confidence (C_prev): {signals['C_prev']:.3f}.")
            if signals.get("static_issues"):
                hints.append("Static issues to fix: " + str(signals["static_issues"]))
            if hints:
                tail.append("\n".join(hints))

        if tail:
            self.core_msg = self.core_msg + "\n\n" + "\n".join(tail)

        logging.info(f"[PatchRefiner] core tokens: {calculate_token(self.core_msg)}")

    @retry((NoCodeError,), tries=3, delay=5)
    def run(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Optional[Dict[str, Any]] = None,
        *,
        last_patch: str,
        test_res: str,
        signals: Optional[Dict[str, Any]] = None,
        attempt: int = 0,
        budget_K: int = 5,
        **kwargs,
    ):
        """
        Produce a refined patch using last patch + failing diagnostics (+ optional signals).
        """
        logging.info("## Running PatchRefiner...")
        if pre_agent_resp is None:
            pre_agent_resp = {}

        fixer_prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/fixer.yaml"))
        refine_prompt = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/refine.yaml"))

        if self.core_msg is None:
            self._generate_core_msg(
                info, pre_agent_resp, last_patch=last_patch, test_res=test_res, signals=signals
            )

        # 若 refine.yaml 中有 'refiner' 键，优先使用；否则退回 'fixer'
        refine_tail = refine_prompt.get("refiner") or refine_prompt.get("fixer") or \
            "\nPlease refine your previous patch to address the failure."

        label_key = "labeled" if "locator" in pre_agent_resp else "unlabeled"

        # 可选：把上一轮模型输出作为 assistant message
        assistant_stub = last_patch if last_patch else ""

        reply = self.send_message(
            [
                {"role": "system", "content": fixer_prompts["sys"][label_key]},
                {"role": "user", "content": self.core_msg + f"\n\n[Attempt {attempt+1}/{budget_K}] " + refine_tail},
                {"role": "assistant", "content": assistant_stub},
            ]
        )
        return self.parse_response(reply)









