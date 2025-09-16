import os
import re
import json
import logging
from typing import Optional, Dict, Any
from collections import defaultdict
from retry import retry

from ..utils import read_yaml
from .agent import Agent, RetryError
from ..prompts.tokens import calculate_token, token_limit


def _extract_json_block(text: str) -> Optional[str]:
    """
    Try to extract a JSON object/array from text (handles fenced code blocks).
    """
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return text[start:end + 1]
    return None


class Summarizer(Agent):
    """
    Summarize classes/functions into a structured dictionary.
    Preferred JSON format (model output). Fallback: legacy '~' lines.
    """

    # ---- JSON-first parser ----
    def _parse_json(self, response: str):
        js = _extract_json_block(response)
        if not js:
            return None
        try:
            data = json.loads(js)
        except Exception as e:
            logging.warning(f"JSON parse failed: {e}")
            return None

        result: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # 1) {"classes":[{"name":..., "functions":[...]}]}
        if isinstance(data, dict) and "classes" in data:
            for cls in data.get("classes", []):
                cname = cls.get("name") or "Anonymous"
                for fn in cls.get("functions", []):
                    fname = fn.get("name") or "unknown"
                    params = fn.get("params") or {}
                    returns = fn.get("returns") or ""
                    desc = fn.get("desc") or fn.get("description") or ""
                    result[cname][fname] = {"paras": params, "return_type": returns, "desp": desc}
            return {"aim": result, "exp": "", "ori": response}

        # 2) {"A":{"f":{"params":...,"returns":...,"desc":...}}, ...}
        if isinstance(data, dict):
            for cname, v in data.items():
                if isinstance(v, dict):
                    for fname, spec in v.items():
                        if isinstance(spec, dict):
                            params = spec.get("params") or spec.get("paras") or {}
                            returns = spec.get("returns") or spec.get("return_type") or ""
                            desc = spec.get("desc") or spec.get("description") or spec.get("desp") or ""
                            result[cname][fname] = {"paras": params, "return_type": returns, "desp": desc}
            if result:
                return {"aim": result, "exp": "", "ori": response}

        return None

    # ---- legacy '~' parser (backward compatible) ----
    def _parse_legacy(self, response: str):
        result: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for line in response.splitlines():
            parts = [a for a in line.split("~") if a.strip()]
            if len(parts) < 3:
                continue
            try:
                class_name = parts[0].strip()[1:-1]
                func_name = parts[1].strip()[1:-1]
                params_raw = parts[2].strip()[1:-1] if len(parts) > 2 else ""
                if params_raw:
                    try:
                        parameters = {param.split(":")[0].strip(): param.split(":")[1].strip()
                                      for param in params_raw.split(",")}
                    except Exception:
                        parameters = [p.strip() for p in params_raw.split(",")]
                else:
                    parameters = {}
                return_type = parts[-2].strip()[1:-1] if len(parts) >= 4 else ""
                desp = parts[-1].strip()[1:-1] if len(parts) >= 5 else ""
                result[class_name][func_name] = {"paras": parameters, "return_type": return_type, "desp": desp}
            except Exception:
                logging.debug(f"Legacy parse skip line: {line}")

        if not result:
            raise RetryError("No valid parts!")
        return {"aim": result, "exp": "", "ori": response}

    def parse_response(self, response: str):
        parsed = self._parse_json(response)
        if parsed is not None:
            return parsed
        return self._parse_legacy(response)

    @retry((RetryError,), tries=3, delay=5)
    def run(
        self,
        code: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
        *args, **kwargs
    ):
        logging.info("## Running Summarizer...")
        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/summarizer.yaml"))

        if code is None and info is not None:
            code = info.get("buggy_code", "")

        reply = self.send_message([
            {"role": "system", "content": prompts["sys"]},
            {"role": "user",   "content": "Raw Code:\n" + (code or "") + "\n" + prompts["end"]}
        ])
        return self.parse_response(reply)