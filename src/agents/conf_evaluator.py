# conf_evaluator.py
from __future__ import annotations
import math
import logging
from typing import Dict, Any
from retry import retry

from .agent import Agent


def _clip01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _size_score(lines_changed: int, alpha: float = 0.08) -> float:
    """Prefer smaller patches; monotone decreasing."""
    lines = max(0, int(lines_changed or 0))
    return math.exp(-alpha * lines)  # in (0,1]


class ConfEvaluator(Agent):
    """
    Confidence scorer & decision maker.
    C = w_align*D_aln + w_static*static + w_compile*compile + w_size*size + w_hist*history
    Decision: compare C vs. theta[level] -> accept / escalate / refine
    """

    # 不需要 LLM 也能工作；继承 Agent 仅为了与框架口径一致
    @retry((Exception,), tries=1, delay=0)
    def run(
        self,
        signals: Dict[str, Any],
        *,
        level: int,
        thresholds: Dict[int | str, float],
        attempt: int = 0,
        budget_K: int | None = None,
        weights: Dict[str, float] | None = None,
        **kwargs,
    ):
        """
        signals: {
          "D_aln": float in [0,1],
          "static_ok": bool|float,
          "compile_ok": bool|float,
          "lines_changed": int,
          "history_ratio": float in [0,1],        # optional (client success rate)
          ...
        }
        """
        w = {"align": 0.50, "static": 0.15, "compile": 0.20, "size": 0.10, "hist": 0.05}
        if weights:
            w.update(weights)

        d_aln = _clip01(signals.get("D_aln", 0.0))
        static_ok = signals.get("static_ok", 0.0)
        static_score = _clip01(1.0 if isinstance(static_ok, bool) and static_ok else static_ok)
        compile_ok = signals.get("compile_ok", 0.0)
        compile_score = _clip01(1.0 if isinstance(compile_ok, bool) and compile_ok else compile_ok)
        size = _size_score(signals.get("lines_changed", 0))
        hist = _clip01(signals.get("history_ratio", 0.0))

        C = (
            w["align"] * d_aln
            + w["static"] * static_score
            + w["compile"] * compile_score
            + w["size"] * size
            + w["hist"] * hist
        )
        C = _clip01(C)

        # 阈值读取（支持 str/int key）
        theta = None
        if level in thresholds:
            theta = thresholds[level]
        elif str(level) in thresholds:
            theta = thresholds[str(level)]
        else:
            raise ValueError(f"Missing threshold for level={level}")

        # 决策：L1/L2 -> escalate；L3 -> refine（直到预算耗尽）
        if C >= theta:
            decision = "accept"
        else:
            if level == 3:
                if budget_K is not None and attempt + 1 >= budget_K:
                    decision = "stop"  # 预算耗尽
                else:
                    decision = "refine"
            else:
                decision = "escalate"

        explain = (
            f"C = {w['align']:.2f}*D_aln({d_aln:.3f}) + {w['static']:.2f}*static({static_score:.3f}) + "
            f"{w['compile']:.2f}*compile({compile_score:.3f}) + {w['size']:.2f}*size({size:.3f}) + "
            f"{w['hist']:.2f}*history({hist:.3f}) = {C:.3f} vs θ_{level}={theta:.3f} -> {decision}"
        )

        payload = {
            "aim": C,
            "exp": explain,
            "ori": "",
            "metrics": {
                "C": C,
                "theta": theta,
                "weights": w,
                "parts": {
                    "D_aln": d_aln,
                    "static": static_score,
                    "compile": compile_score,
                    "size": size,
                    "history": hist,
                },
                "level": level,
                "attempt": attempt,
                "budget_K": budget_K,
            },
            "decision": decision,
        }
        return payload
