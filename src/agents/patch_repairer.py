import os
import logging
from typing import Optional, Dict, Any
from retry import retry

from ..utils import read_yaml
from ..parse import parse_code, parse_exp
from ..prompts.tokens import calculate_token, token_limit
from .agent import Agent, NoCodeError


class PatchRepairer(Agent):
    """
    Former 'Fixer'. Generates initial patches (L1/L2) given anchor/contexts.
    - supports labeled (with locator) and unlabeled modes
    - can consume optional 'signals' (e.g., D_aln, C_prev) to steer prompts
    """

    def parse_response(self, response: str):
        parts = parse_code(response)
        if not parts:
            raise NoCodeError("No code block found in model response.")
        patch = parts[0].strip()
        if "===" in patch:
            patch = patch[: patch.find("===")].strip()
        return {"aim": patch, "exp": parse_exp(response), "ori": response}

    # ---- message assembly ----
    def _generate_core_msg(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Dict[str, Any],
        signals: Optional[Dict[str, Any]] = None,
    ):
        if "locator" in pre_agent_resp:
            logging.info("Fix code with marks of buggy lines")
            self.core_msg = (
                "The following code contains a bug with suspicious lines labeled:\n"
                + pre_agent_resp["locator"]
            )
        else:
            self.core_msg = "The following code contains a bug:\n" + info["buggy_code"]

        # common shared context (tests, summaries, helper, ...)
        self._shared_msg(info, pre_agent_resp)

        # optional coverage
        if "coverage_report" in info and calculate_token(self.core_msg + info["coverage_report"]) <= token_limit[self.model_name]["overall"]:
            self.core_msg = "Code coverage for failed testcases:\n" + info["coverage_report"] + "\n" + self.core_msg

        # optional alignment/confidence signals to nudge the model
        if signals:
            hints = []
            if "D_aln" in signals:
                hints.append(f"Current code–text alignment score (D_aln): {signals['D_aln']:.3f}.")
                hints.append("Please improve consistency with textual intent when revising the patch.")
            if "C_prev" in signals:
                hints.append(f"Previous attempt confidence (C_prev): {signals['C_prev']:.3f}.")
            if hints:
                self.core_msg = "\n".join(hints) + "\n" + self.core_msg

        logging.info(f"Current core message tokens: {calculate_token(self.core_msg)}")

    # ---- main entry ----
    @retry((NoCodeError,), tries=3, delay=5)
    def run(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Optional[Dict[str, Any]] = None,
        *,
        level: Optional[int] = None,
        signals: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        logging.info("## Running PatchRepairer...")
        if pre_agent_resp is None:
            pre_agent_resp = {}

        label_key = "labeled" if "locator" in pre_agent_resp else "unlabeled"
        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/fixer.yaml"))

        if self.core_msg is None:
            self._generate_core_msg(info, pre_agent_resp, signals=signals)

        reply = self.send_message([
            {"role": "system", "content": prompts["sys"][label_key]},
            {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]}
        ])
        return self.parse_response(reply)

    # ---- refinement (used by L3 or failed attempts) ----
    def refine(
        self,
        assist_resp: str,
        test_res: str,
        *,
        signals: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        refine_prompt = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/refine.yaml"))

        # optional signal-aware nudge
        tail_hint = ""
        if signals and "D_aln" in signals:
            tail_hint = f"\nAlignment score (D_aln): {signals['D_aln']:.3f}. Align the fix with textual intent."

        fixer_prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/fixer.yaml"))
        label_key = "labeled" if ("locator" in (signals or {})) else "unlabeled"

        reply = self.send_message([
            {"role": "system",    "content": fixer_prompts["sys"][label_key]},
            {"role": "user",      "content": self.core_msg + "\n" + refine_prompt.get("prefix", "")},
            {"role": "assistant", "content": assist_resp},
            {"role": "user",      "content": "\nYour generated patch fails because:\n" + test_res + tail_hint + refine_prompt.get("fixer", "")}
        ])
        return self.parse_response(reply)







