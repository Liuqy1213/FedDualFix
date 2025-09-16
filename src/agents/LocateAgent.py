import logging
import os
from typing import Optional, Dict, Any
from retry import retry

from .agent import Agent, RetryError, NoCodeError
from ..prompts.tokens import calculate_token, token_limit
from ..utils import read_yaml
from ..parse import parse_code, parse_exp, exist_line, unique_matching, search_valid_line


COMMENT_BY_LANG = {
    "java": "//",
    "c": "//",
    "cpp": "//",
    "c++": "//",
    "cs": "//",
    "csharp": "//",
    "go": "//",
    "js": "//",
    "ts": "//",
    "kotlin": "//",
    "scala": "//",
    "swift": "//",
    "python": "#",
    "py": "#",
    "ruby": "#",
    "rs": "//",   # rust
}

def _comment_label_from(info: Dict[str, Any]) -> str:
    lang = (info.get("project_meta", {}).get("language") or "").lower()
    return COMMENT_BY_LANG.get(lang, "//")


class LocateAgent(Agent):
    """Line-level localization with (optional) code slice context."""

    def parse_response(self, response: str, raw_code: str, comment_label: str = "//"):
        resp_code = "\n".join([p.strip() for p in parse_code(response) if p.strip()])
        if "===" in resp_code:
            resp_code = resp_code[:resp_code.find("===")].strip()

        resp_lines, raw_code_lines = resp_code.splitlines(), raw_code.splitlines()
        raw_lines_w_marks = [l for l in raw_code_lines]

        mark_indces = {i: (-1, "") for i, l in enumerate(resp_lines) if "missing" in l or "buggy" in l}
        for i, l in enumerate(resp_lines):
            if i not in mark_indces and i - 1 in mark_indces and not exist_line(l, raw_code_lines):
                mark_indces[i] = (-2, "")

        if len(mark_indces) == 0:
            raise RetryError("No mark in the response!")

        for resp_idx in sorted(list(mark_indces.keys())):
            if mark_indces[resp_idx][0] == -2 and resp_idx - 1 in mark_indces and mark_indces[resp_idx - 1][0] >= 0:
                mark_indces[resp_idx] = mark_indces[resp_idx - 1]
                (unique_idx, mode) = mark_indces[resp_idx - 1]
                if mode == "pre":
                    raw_lines_w_marks[unique_idx] += f"\n/* missing code:[{resp_lines[resp_idx].rstrip()}] */"
                else:
                    this_lines = raw_lines_w_marks[unique_idx].split("\n")
                    added, ori = "\n".join(this_lines[:-1]), this_lines[-1]
                    raw_lines_w_marks[unique_idx] = added + f"\n/* missing code:[{resp_lines[resp_idx]}] */\n" + ori
                continue

            parts = resp_lines[resp_idx].split(comment_label, 1)
            code = parts[0].rstrip()
            comment = parts[1].strip() if len(parts) > 1 else "buggy line"

            unique_idx = unique_matching(resp_lines, raw_code_lines, resp_idx)
            if unique_idx >= 0:
                raw_lines_w_marks[unique_idx] += f" {comment_label} " + comment
                mark_indces[resp_idx] = (unique_idx, "pre")
                continue
            elif unique_idx == -2 and (len(code) or "missing" in resp_lines[resp_idx]) > 0:
                pre_valid = search_valid_line(resp_lines, resp_idx, "pre", existing=raw_code_lines)
                if pre_valid is not None:
                    unique_idx = unique_matching(resp_lines, raw_code_lines, pre_valid[0], existing=True)
                    if unique_idx >= 0:
                        raw_lines_w_marks[unique_idx] += (f"\n/* missing code:[{code}] // {comment} */")
                        mark_indces[resp_idx] = (unique_idx, "pre")
                        continue
                post_valid = search_valid_line(resp_lines, resp_idx, "post", existing=raw_code_lines)
                if post_valid is not None:
                    unique_idx = unique_matching(resp_lines, raw_code_lines, post_valid[0], existing=True)
                    if unique_idx >= 0:
                        raw_lines_w_marks[unique_idx] = (f"/* missing code:[{code}] // {comment} */\n") + raw_lines_w_marks[unique_idx]
                        mark_indces[resp_idx] = (unique_idx, "post")

        success = sum([i[0] >= 0 for i in mark_indces.values()])
        if success == 0:
            raise RetryError(f"Cannot mark any line with {len(mark_indces)} marks")

        if success < len(mark_indces):
            logging.warning("Some labeled lines seem not from the original code")
            for mark_idx in mark_indces:
                if not mark_indces[mark_idx]:
                    resp_lines[mark_idx] += f"  {comment_label} Cannot Mark!"

        return {"aim": "\n".join(raw_lines_w_marks), "exp": parse_exp(response), "ori": response}

    def _generate_core_msg(self, info: Dict[str, Any], pre_agent_resp: Dict[str, Any]):
        if "slicer" in pre_agent_resp:
            logging.info("Mark buggy lines on suspicious code segment")
            self.core_msg = "The following code contains a bug:\n" + pre_agent_resp["slicer"]
        else:
            self.core_msg = "The following code contains a bug:\n" + info["buggy_code"]

        self._shared_msg(info, pre_agent_resp)
        logging.info(f"Current core message tokens: {calculate_token(self.core_msg)}")

    def fast_parse(self, response: str):
        resp_code = "\n".join([p.strip() for p in parse_code(response) if p.strip()])
        if "===" in resp_code:
            resp_code = resp_code[:resp_code.find("===")].strip()
        return {"aim": resp_code, "exp": parse_exp(response), "ori": response}

    @retry((NoCodeError,), tries=3, delay=5)
    def run(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Optional[Dict[str, Any]] = None,
        *,
        level: Optional[int] = None,
        signals: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        **kwargs
    ):
        logging.info("## Running LocateAgent...")
        if pre_agent_resp is None:
            pre_agent_resp = {}

        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/locator.yaml"))
        if self.core_msg is None:
            self._generate_core_msg(info, pre_agent_resp)

        raw_code = pre_agent_resp.get("slicer", info["buggy_code"])
        comment_label = _comment_label_from(info)

        attempt = 0
        bk_resp = None
        while attempt < max_retries:
            try:
                response = self.send_message([
                    {"role": "system", "content": prompts["sys"]},
                    {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]}
                ])
                return self.parse_response(response, raw_code, comment_label=comment_label)

            except NoCodeError:
                attempt += 1
                logging.warning("No code, try again")
            except RetryError:
                attempt += 1
                mark = sum([("// buggy line" in l or "// missing" in l) for l in parse_code(response)])
                if mark > 0:
                    bk_resp = response
                else:
                    logging.warning("Cannot mark any line, try again")

        if bk_resp is not None:
            return self.fast_parse(bk_resp)
        else:
            raise ValueError("No available localization results!")







