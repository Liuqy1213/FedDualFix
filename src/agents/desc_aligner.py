# desc_aligner.py
from __future__ import annotations
import os
import re
import math
import json
import logging
from typing import Dict, List, Optional
from retry import retry

from .agent import Agent, RetryError


def _norm01(x: float) -> float:
    if x is None or math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _cosine(u: List[float], v: List[float]) -> float:
    import numpy as np
    a, b = np.array(u, dtype=float), np.array(v, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _bow_vec(text: str) -> List[float]:
    # very light bag-of-words vector for fallback cosine
    from collections import Counter
    toks = re.findall(r"[A-Za-z_]+", (text or "").lower())
    cnt = Counter(toks)
    keys = sorted(cnt.keys())
    return [cnt[k] for k in keys]


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def grams(s: str) -> set:
        s = re.sub(r"\s+", " ", s.strip())
        return set([s[i:i+n] for i in range(max(0, len(s)-n+1))])
    A, B = grams(a or ""), grams(b or "")
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


class DescAligner(Agent):
    """
    Compute D_aln using three signals: NLI entailment, CodeBLEU (code-code), and cross-modal cosine.
    """

    def _embed_text(self, text: str, model: Optional[str]) -> List[float]:
        try:
            # OpenAI embeddings path (preferred)
            if hasattr(self.client, "embeddings") and callable(getattr(self.client, "embeddings").create):
                emb_model = model or "text-embedding-3-large"
                vec = self.client.embeddings.create(model=emb_model, input=(text or ""))  # type: ignore
                return vec.data[0].embedding  # type: ignore
        except Exception as e:
            logging.debug(f"[DescAligner] embed fallback: {e}")
        # fallback to bag-of-words
        return _bow_vec(text or "")

    def _codebleu(self, g: str, g_ctx: str) -> float:
        # try real CodeBLEU if available
        try:
            from codebleu import calc_code_bleu  # type: ignore
            # Guess language; users可在 info 里传 language 进一步覆盖
            lang = "java"
            refs = [g_ctx or ""]
            hyp = g or ""
            score_dict = calc_code_bleu.get_codebleu(refs, hyp, lang)
            # codebleu库可能返回 dict 或 tuple
            if isinstance(score_dict, dict):
                return float(score_dict.get("codebleu", 0.0))
            if isinstance(score_dict, (list, tuple)) and score_dict:
                return float(score_dict[0])
        except Exception as e:
            logging.debug(f"[DescAligner] CodeBLEU fallback: {e}")
        # fallback: n-gram jaccard
        return _ngram_jaccard(g or "", g_ctx or "", n=3)

    def _nli_entail(self, premise: str, hypothesis: str) -> float:
        """
        Return entailment score in [0,1].
        Preferred: LLM classification; fallback: keyword overlap.
        """
        try:
            prompt = (
                "You are a precise NLI classifier. "
                "Label the relation between PREMISE and HYPOTHESIS as one of: entailment, neutral, contradiction.\n"
                f"PREMISE:\n{premise}\n\nHYPOTHESIS:\n{hypothesis}\n"
                "Answer with a single word: entailment/neutral/contradiction."
            )
            resp = self.send_message([
                {"role": "system", "content": "You are a helpful NLI classifier."},
                {"role": "user", "content": prompt}
            ])
            text = (resp or "").lower()
            if "entail" in text:
                return 1.0
            if "contradict" in text:
                return 0.0
            return 0.5
        except Exception as e:
            logging.debug(f"[DescAligner] NLI fallback: {e}")
            # fallback: simple token overlap
            p = set(re.findall(r"[A-Za-z_]+", (premise or "").lower()))
            h = set(re.findall(r"[A-Za-z_]+", (hypothesis or "").lower()))
            return _norm01(len(p & h) / (len(h) + 1e-6))

    def _sigma(self, patch_text: str) -> str:
        """
        Diff-to-text verbalization: extract '+' lines or code identifiers as a concise summary.
        """
        lines = (patch_text or "").splitlines()
        added = [ln for ln in lines if ln.strip().startswith("+")]
        if added:
            return "\n".join(added[:20])
        # fallback: keep first N non-empty lines
        keep = [ln for ln in lines if ln.strip()]
        return "\n".join(keep[:20])

    @retry((RetryError,), tries=3, delay=5)
    def run(self,
            patch: str,
            texts: str,
            context_code: str,
            *,
            lambdas: Dict[str, float] | None = None,
            embed_model: Optional[str] = None,
            info: Optional[dict] = None,
            **kwargs):
        """
        Compute D_aln for (g=patch, T=texts, g_ctx=context_code).
        """
        lam = {"e": 0.45, "b": 0.35, "c": 0.20}
        if isinstance(lambdas, dict):
            lam.update({k[0]: float(v) for k, v in lambdas.items()})  # accept {"lambda_e":0.5,...} or {"e":0.5,...}

        # 1) NLI entailment on sigma(g) vs T
        sigma_g = self._sigma(patch)
        nli = _norm01(self._nli_entail(sigma_g, texts))

        # 2) CodeBLEU on code-code
        codebleu = _norm01(self._codebleu(patch, context_code))

        # 3) cross-modal cosine via embeddings (or BoW fallback)
        vec_g = self._embed_text(patch, model=embed_model)
        vec_t = self._embed_text(texts, model=embed_model)
        cos = _norm01(_cosine(vec_g, vec_t))

        d_aln = lam["e"] * nli + lam["b"] * codebleu + lam["c"] * cos

        explain = (
            f"D_aln = λ_e*NLI + λ_b*CodeBLEU + λ_c*cos = "
            f"{lam['e']:.2f}*{nli:.3f} + {lam['b']:.2f}*{codebleu:.3f} + {lam['c']:.2f}*{cos:.3f} = {d_aln:.3f}"
        )

        payload = {
            "aim": d_aln,
            "exp": explain,
            "ori": json.dumps({"sigma_g": sigma_g[:200]}),
            "metrics": {
                "NLI": nli,
                "CodeBLEU": codebleu,
                "cos": cos,
                "lambda": {"e": lam["e"], "b": lam["b"], "c": lam["c"]},
            },
        }
        return payload
