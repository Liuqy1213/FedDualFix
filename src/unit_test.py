# unit_test.py
import os
import sys
import json
import logging
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import json_pretty_dump, read_json, return_lines
from src.prompts.prepare import get_info_dict

def test_context(info):
    from src.agents.context_agent import ContextAgent
    agent = ContextAgent("gpt-4o", hash_id="unittest")
    resp = agent.run(info)
    _ensure_dirs()
    _save_ori("context", resp["ori"])
    _save_aim("context", resp["aim"]["texts"] if isinstance(resp["aim"], dict) else str(resp["aim"]))
    print("[context] done.")

def test_focus(model_name, info):
    from src.agents.FocusAgent import FocusAgent
    agent = FocusAgent(model_name, hash_id="unittest")
    resp = agent.run(info)
    _ensure_dirs()
    _save_ori("focus", resp["ori"])
    _save_aim("focus", "\n".join(resp["aim"]))
    print("[focus] files:", *resp["aim"], sep="\n  - ")

def test_summarizer(model_name, info):
    from src.agents.summarizer import Summarizer
    bug_related = return_lines("../unit/aim_save/focus.txt")
    agent = Summarizer(model_name, hash_id="unittest")
    summary = {}
    for f in bug_related:
        if not f.endswith(".java"):
            continue
        fpath = os.path.join(info["project_meta"]["project_src_path"], f)
        if os.path.exists(fpath):
            with open(fpath, encoding="utf-8") as rf:
                code = rf.read()
            resp = agent.run(code)
            summary[f] = resp["aim"]
    _ensure_dirs()
    json_pretty_dump(summary, "../unit/aim_save/summarizer.json")
    _save_ori("summarizer", json.dumps(summary, ensure_ascii=False))
    print(f"[summarizer] summarized {len(summary)} files")

def test_slicer(model_name, info):
    from src.agents.slicer import SliceAgent
    agent = SliceAgent(model_name, hash_id="unittest")

    pre = {}
    if os.path.exists("../unit/aim_save/context.txt"):
        with open("../unit/aim_save/context.txt", encoding="utf-8") as rf:
            pre["helper"] = rf.read()
    if os.path.exists("../unit/aim_save/summarizer.json"):
        pre["summarizer"] = json.dumps(read_json("../unit/aim_save/summarizer.json"))

    resp = agent.run(info, pre_agent_resp=pre)
    _ensure_dirs()
    _save_ori("slicer", resp["ori"])
    _save_aim("slicer", resp["aim"], ext="java")
    print("[slicer] span:", resp.get("metrics"))

def test_locator(model_name, info):
    from src.agents.LocateAgent import LocateAgent
    agent = LocateAgent(model_name, hash_id="unittest")

    pre = {}
    slicer_path = "../unit/aim_save/slicer.java"
    if os.path.exists(slicer_path):
        with open(slicer_path, encoding="utf-8") as rf:
            pre["slicer"] = rf.read()

    if os.path.exists("../unit/aim_save/context.txt"):
        with open("../unit/aim_save/context.txt", encoding="utf-8") as rf:
            pre["helper"] = rf.read()
    if os.path.exists("../unit/aim_save/summarizer.json"):
        pre["summarizer"] = json.dumps(read_json("../unit/aim_save/summarizer.json"))

    resp = agent.run(info, pre_agent_resp=pre)
    _ensure_dirs()
    _save_ori("locator", resp["ori"])
    _save_aim("locator", resp["aim"], ext="java")
    print("[locator] done.")

def test_patch(model_name, info):
    from src.agents.patch_repairer import PatchRepairer
    agent = PatchRepairer(model_name, hash_id="unittest")

    pre = {}
    if os.path.exists("../unit/aim_save/locator.java"):
        with open("../unit/aim_save/locator.java", encoding="utf-8") as rf:
            pre["locator"] = rf.read()
    if os.path.exists("../unit/aim_save/context.txt"):
        with open("../unit/aim_save/context.txt", encoding="utf-8") as rf:
            pre["helper"] = rf.read()
    if os.path.exists("../unit/aim_save/summarizer.json"):
        pre["summarizer"] = json.dumps(read_json("../unit/aim_save/summarizer.json"))

    resp = agent.run(info, pre_agent_resp=pre)
    _ensure_dirs()
    _save_ori("patch", resp["ori"])
    _save_aim("patch", resp["aim"], ext="patch")
    print("[patch] patch saved.")

def test_refiner(model_name, info, container_id):
    from src.agents.patch_refiner import PatchRefiner
    from src.patch import patching_and_testing

    patch_path = "../unit/aim_save/patch.patch"
    if not os.path.exists(patch_path):
        raise RuntimeError("No patch found. Run --test patch first.")
    with open(patch_path, encoding="utf-8") as rf:
        last_patch = rf.read()

    # 准备必要的 meta（示例）
    info["project_meta"]["checkout_dir"] = "checkouts"
    info["project_meta"]["buggy_file_path"] = "checkouts/Lang_1_buggy/src/main/java/org/apache/commons/lang3/math/NumberUtils.java"

    plausible = patching_and_testing(last_patch, info["project_meta"], container_id)
    print(f"[refiner] previous patch plausible? {plausible}")

    pre = {}
    if os.path.exists("../unit/aim_save/context.txt"):
        with open("../unit/aim_save/context.txt", encoding="utf-8") as rf:
            pre["helper"] = rf.read()
    if os.path.exists("../unit/aim_save/summarizer.json"):
        pre["summarizer"] = json.dumps(read_json("../unit/aim_save/summarizer.json"))

    agent = PatchRefiner(model_name, hash_id="unittest")
    resp = agent.run(
        info,
        pre_agent_resp=pre,
        last_patch=last_patch,
        test_res="" if plausible else "Some tests still failing.",
        signals={"C_prev": 1.0 if plausible else 0.2},
        attempt=0,
        budget_K=5,
    )
    _ensure_dirs()
    _save_ori("refiner", resp["ori"])
    _save_aim("refiner", resp["aim"], ext="patch")
    print("[refiner] refined patch saved.")

# ---------------- util for unit outputs ----------------

def _ensure_dirs():
    os.makedirs("../unit/ori_resp", exist_ok=True)
    os.makedirs("../unit/aim_save", exist_ok=True)

def _save_ori(name: str, text: str):
    with open(f"../unit/ori_resp/{name}.txt", "w", encoding="utf-8") as wf:
        wf.write(text)

def _save_aim(name: str, text: str, ext: str = "txt"):
    with open(f"../unit/aim_save/{name}.{ext}", "w", encoding="utf-8") as wf:
        wf.write(text)

# ---------------- main ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="slicer",
                        choices=["context", "focus", "summarizer", "slicer", "locator", "patch", "refiner"])
    parser.add_argument("--model_name", default="gpt-4o")
    parser.add_argument("--container_id", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[logging.FileHandler(f"../unit/{args.test}.log"), logging.StreamHandler()],
    )

    info = get_info_dict(
        checkout_dir="../unit/cases",
        bug_name="Lang_1",
        model_name=args.model_name,
        root_causes={"Lang_1": "src/main/java/org/apache/commons/lang3/math/NumberUtils.java"},
    )

    if args.test == "context":    test_context(info)
    if args.test == "focus":      test_focus(args.model_name, info)
    if args.test == "summarizer": test_summarizer(args.model_name, info)
    if args.test == "slicer":     test_slicer(args.model_name, info)
    if args.test == "locator":    test_locator(args.model_name, info)
    if args.test == "patch":      test_patch(args.model_name, info)
    if args.test == "refiner":    test_refiner(args.model_name, info, args.container_id)






