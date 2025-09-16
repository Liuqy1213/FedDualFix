# context_agent.py
import os
import re
import json
import logging
from typing import Optional, Dict, Any
from retry import retry

from ..utils import read_yaml
from .agent import Agent, RetryError
from ..prompts.tokens import calculate_token, token_limit

try:
    from tavily import TavilyClient
    _HAS_TAVILY = True
except Exception:
    _HAS_TAVILY = False


def _maybe_tavily_search(query: str, base_dir: str) -> Optional[str]:
    """
    Try to run Tavily search if env is configured; otherwise return None.
    """
    if not _HAS_TAVILY:
        return None
    env_path = os.path.join(base_dir, "tavily_env.yaml")
    if not os.path.exists(env_path):
        return None
    tavily_env = read_yaml(env_path)
    api_key = tavily_env.get("api_key")
    if not api_key:
        return None
    client = TavilyClient(api_key=api_key)
    try:
        return client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    except Exception as e:
        logging.warning(f"Tavily search failed: {e}")
        return None


def _extract_json_block(text: str) -> Optional[str]:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback to braces
    s, e = text.find("{"), text.rfind("}")
    if 0 <= s < e:
        return text[s: e + 1]
    return None


class ContextAgent(Agent):
    """
    Aggregate project context (bug report, commit message, docstrings, prior fixes, optional web hints).
    Output: structured dict with 'meta' and 'texts'.
    """

    def parse_response(self, response: str, info_meta: Dict[str, Any]):
        """
        Prefer JSON; fallback to plain text block as 'texts'.
        Expected JSON shape (flexible):
        {
          "notes": "...",
          "hints": ["...", "..."],
          "citations": [{"query": "...", "source": "..."}]
        }
        """
        js = _extract_json_block(response)
        payload = {"meta": info_meta, "texts": "", "notes": "", "hints": [], "citations": []}

        if js:
            try:
                obj = json.loads(js)
                if isinstance(obj, dict):
                    payload["notes"] = obj.get("notes", "")
                    payload["hints"] = obj.get("hints", []) or obj.get("bullets", [])
                    payload["citations"] = obj.get("citations", [])
                    if "texts" in obj:
                        payload["texts"] = obj["texts"]
                else:
                    payload["notes"] = str(obj)
            except Exception as e:
                logging.warning(f"[ContextAgent] JSON parse failed: {e}")

        if not payload["texts"]:
            payload["texts"] = response.strip()

        if not (payload["texts"] or payload["notes"] or payload["hints"]):
            raise RetryError("Empty context payload.")

        return {"aim": payload, "exp": "", "ori": response}

    def _generate_core_msg(self, info: Dict[str, Any], pre_agent_resp: Dict[str, Any]):
        """
        Assemble the seed prompt: bug info + optional artifacts.
        """
        parts = []
        if info.get("bug_report"):
            parts.append("Bug report:\n" + info["bug_report"])
        if info.get("commit_msg"):
            parts.append("Related commit message:\n" + info["commit_msg"])
        if info.get("docstrings"):
            parts.append("Relevant docstrings:\n" + info["docstrings"])
        if info.get("history_fixes"):
            parts.append("Historical fixes (snippets):\n" + info["history_fixes"])

        if info.get("buggy_code"):
            parts.append("Buggy code snippet:\n" + info["buggy_code"])

        self.core_msg = "\n\n".join(parts) if parts else "Context aggregation required for the bug."

        self._shared_msg(info, pre_agent_resp)

        if "coverage_report" in info and calculate_token(self.core_msg + info["coverage_report"]) <= token_limit[self.model_name]["overall"]:
            self.core_msg = "Code coverage for failed testcases:\n" + info["coverage_report"] + "\n" + self.core_msg

        logging.info(f"[ContextAgent] core tokens: {calculate_token(self.core_msg)}")

    @retry((RetryError,), tries=3, delay=5)
    def run(
        self,
        info: Dict[str, Any],
        pre_agent_resp: Optional[Dict[str, Any]] = None,
        *,
        use_web: bool = True,
        max_tries: int = 6,
        **kwargs
    ):
        logging.info("## Running ContextAgent...")
        if pre_agent_resp is None:
            pre_agent_resp = {}

        prompts = read_yaml(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../prompts/helper.yaml"))

        if self.core_msg is None:
            self._generate_core_msg(info, pre_agent_resp)

        # tools: Tavily (optional)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        tools = None
        if use_web and _HAS_TAVILY and os.path.exists(os.path.join(base_dir, "tavily_env.yaml")):
            tools = [{
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "Get related solutions to fix the bug from the web.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query describing the error/context, <= 100 words."
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    }
                }
            }]

        if tools:
            # tool-use path
            for _ in range(max_tries):
                resp = self.send_message(
                    msg=[
                        {"role": "system", "content": prompts["sys"]},
                        {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]},
                    ],
                    tools=tools,
                    handling=False
                )
                if getattr(resp.choices[0], "finish_reason", "") == "tool_calls":
                    arguments = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
                    query = (arguments or {}).get("query", "")
                    logging.info("[ContextAgent] Web query: %s", query)
                    context = _maybe_tavily_search(query, base_dir) or ""
                    final = self.send_message([
                        {"role": "system", "content": prompts["sys"]},
                        {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]},
                        resp.choices[0].message,
                        {"role": "tool",
                         "content": json.dumps({"query": query, "tavily_search_result": context}),
                         "tool_call_id": resp.choices[0].message.tool_calls[0].id}
                    ])
                    info_meta = {
                        "project": info.get("project_meta", {}),
                        "have_web": bool(context),
                    }
                    return self.parse_response(final, info_meta=info_meta)

        # no-tool fallback (or Tavily unavailable)
        final = self.send_message([
            {"role": "system", "content": prompts["sys"]},
            {"role": "user",   "content": self.core_msg + "\n" + prompts["end"]}
        ])
        info_meta = {
            "project": info.get("project_meta", {}),
            "have_web": False,
        }
        return self.parse_response(final, info_meta=info_meta)





