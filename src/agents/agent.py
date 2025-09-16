from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Any
import logging
from retry import retry

import openai
import google.generativeai as genai

from ..utils import read_json
from ..prompts.tokens import calculate_token, token_limit


class NoCodeError(Exception):
    def __init__(self, message: str = "No code found"):
        super().__init__(message)


class RetryError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class Agent(ABC):
    """
    Base class for all agents.
    Changes vs. original:
      - set_client -> _set_client, remove hardcoded proxy; read base_url/api_key from config if provided
      - __shared_msg -> _shared_msg (protected, extensible)
      - add decoding params & score_hooks to support confidence/alignment scoring
      - send_message supports tools; applies hooks after getting text
    """

    def __init__(
        self,
        model_name: str,
        hash_id: str,
        config_path: str = "../config.json",
        *,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.hash_id = hash_id
        self.client = self._set_client(config_path)
        self.core_msg: Optional[str] = None

        # decoding params (used for OpenAI-style clients)
        self.decoding = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)

        # optional score hooks: List[Callable[[Agent, str], None]]
        self.score_hooks: List[Callable[["Agent", str], None]] = []

    def __str__(self) -> str:
        return self.__class__.__name__

    __repr__ = __str__

    # ---------- client & messaging ----------

    def _set_client(self, config_path: str):
        """Create a provider client from config. No hardcoded proxies."""
        config = read_json(config_path)

        # Prefer explicit keys; fallbacks keep compatibility
        openai_key = config.get("OpenAI") or config.get("ChatGPT")
        openai_base = config.get("OpenAI_BASE")  # optional

        if self.model_name.startswith(("gpt", "claude", "o")):
            kwargs = {}
            if openai_base:
                kwargs["base_url"] = openai_base
            if openai_key:
                kwargs["api_key"] = openai_key
            return openai.OpenAI(**kwargs)

        if self.model_name.startswith("deepseek"):
            return openai.OpenAI(base_url="https://api.deepseek.com/v1", api_key=config["DeepSeek"])

        if self.model_name.startswith("Phind"):
            return openai.OpenAI(base_url="https://api.deepinfra.com/v1/openai", api_key=config["DeepInfra"])

        if self.model_name.startswith("gemini"):
            genai.configure(api_key=config["Gemini"])
            return genai.GenerativeModel(
                self.model_name,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ],
            )

        # default fallback to OpenAI SDK with env var credentials
        return openai.OpenAI()

    def _handle_openai_response(self, response):
        finish = response.choices[0].finish_reason
        text = response.choices[0].message.content or ""
        logging.debug(f"[LLM finish_reason] {finish}")
        if finish in {"stop", "length"}:
            if finish == "length":
                logging.warning(f"Response truncated with {len(text)} characters.")
            return text
        if finish == "tool_calls":
            return response  # caller handles tools
        if finish == "content_filter":
            logging.warning("Content filtered by provider.")
            return None
        raise RetryError("try again")

    def _handle_gemini_response(self, response):
        status = response.candidates[0].finish_reason._name_
        text = response.text or ""
        logging.debug(f"[Gemini finish_reason] {status}")
        return text if status == "STOP" else response

    def _dict_prompt_to_text(self, msg: List[Dict[str, str]]) -> str:
        return "\n".join([d["content"] for d in msg])

    # ---------- shared message assembly (extensible) ----------

    def _shared_msg(self, info: Dict, pre_agent_resp: Dict) -> None:
        """
        Assemble shared context into self.core_msg.
        Subclasses may call this and then append their own signals.
        """
        # failing tests
        if info.get("failing_test_cases"):
            self.core_msg = (self.core_msg or "") + "\nThe code fails on this test:\n" + info["failing_test_cases"]

        # summarizer/context/helper (respect token budget)
        if "summarizer" in pre_agent_resp:
            if calculate_token((self.core_msg or "") + pre_agent_resp["summarizer"]) <= token_limit[self.model_name]["overall"]:
                self.core_msg = "Related code summary:\n" + pre_agent_resp["summarizer"] + "\n" + (self.core_msg or "")

        if "helper" in pre_agent_resp:
            if calculate_token((self.core_msg or "") + pre_agent_resp["helper"]) <= token_limit[self.model_name]["overall"]:
                self.core_msg = "Reference debugging guide:\n" + pre_agent_resp["helper"] + "\n" + (self.core_msg or "")

    # ---------- hooks ----------

    def register_score_hook(self, fn: Callable[[ "Agent", str ], None]) -> None:
        """Register a post-generation scoring hook (e.g., compute D_aln / C)."""
        self.score_hooks.append(fn)

    # ---------- chat ----------

    @retry((RetryError, Exception), tries=6, delay=30, backoff=2)
    def send_message(self, msg: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, handling: bool = True):
        try:
            if not self.model_name.startswith("gemini"):
                kwargs = dict(model=self.model_name, messages=msg)
                kwargs.update(self.decoding)
                if tools:
                    kwargs["tools"] = tools
                resp = self.client.chat.completions.create(**kwargs)
                out = self._handle_openai_response(resp) if handling else resp
            else:
                resp = self.client.generate_content(self._dict_prompt_to_text(msg))
                out = self._handle_gemini_response(resp) if handling else resp

            # apply hooks on text output
            if handling and isinstance(out, str):
                for fn in self.score_hooks:
                    try:
                        fn(self, out)
                    except Exception as hook_err:
                        logging.debug(f"[score_hook skipped] {hook_err}")

            return out

        except Exception as e:
            logging.warning(f"[Retry Triggered] Exception in send_message: {e}")
            raise RetryError(str(e))

    # ---------- subclass API ----------

    @abstractmethod
    def parse_response(self, response: str, *args):
        ...

    @abstractmethod
    def run(self, info: Dict, *args):
        ...