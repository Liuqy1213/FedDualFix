
import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.patch_repairer import PatchRepairer
from src.agents.LocateAgent import LocateAgent
from src.agents.slicer import SliceAgent
from src.agents.summarizer import Summarizer
from src.agents.patch_refiner import PatchRefiner
from src.agents.FocusAgent import FocusAgent
from src.agents.context_agent import ContextAgent

from src.prompts.prepare import get_info_dict
from src.prompts.tokens import token_limit
from src.patch import patching_and_testing
from src.utils import dump_exp, return_lines, write_line, json_pretty_dump, read_json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", default=1, type=int)
parser.add_argument("--model_name", default="gpt-4o", choices=list(token_limit.keys()))
parser.add_argument("--data_name", default="d4j")
parser.add_argument("--level", default=3, type=int, choices=[1, 2, 3])
parser.add_argument("--container_id", default=None, type=str)
parser.add_argument("--re_patch_num", default=2, type=int)
parser.add_argument("--refinement", action="store_true")
# 如需实际运行，把上面这行改为：params = vars(parser.parse_args())
params = vars(parser.parse_args([]))

# 角色映射到类
role_dict = {
    "focus":       FocusAgent,
    "context":     ContextAgent,
    "locator":     LocateAgent,
    "slicer":      SliceAgent,
    "summarizer":  Summarizer,
    "patch":       PatchRepairer,
    "refiner":     PatchRefiner,
}

# 文件后缀（用于保存 aim/exp/ori）
suffix_dict = {
    "focus":      "json",
    "context":    "txt",
    "locator":    "java",
    "slicer":     "java",
    "summarizer": "json",
    "patch":      "patch",
    "refiner":    "patch",
}

# 各层级流水线
level_dict = {
    1: ["locator", "patch"],
    2: ["summarizer", "slicer", "locator", "patch"],
    3: ["context", "focus", "summarizer", "slicer", "locator", "patch", "refiner"],
}


class Pipeline:
    def __init__(self, model_name: str, container_id, data_name, refinement=False, level=3, **kwargs):
        self.model_name = model_name
        self.container_id = None if container_id in ["None", None] else container_id
        self.data_name = data_name
        self.level = level
        self.refinement = refinement

        self.record_dir, self.hash_id = dump_exp(
            f"../res/{data_name}/records",
            {"model_name": model_name, "level": level},
        )
        self.framework = {role: role_dict[role](model_name, self.hash_id) for role in level_dict[level]}
        self.response_dir = os.path.join(f"../res/{data_name}/resp", self.hash_id)

        os.makedirs(self.record_dir, exist_ok=True)
        self.records = {
            "worked":    return_lines(os.path.join(self.record_dir, "worked.txt")),
            "failed":    return_lines(os.path.join(self.record_dir, "failed.txt")),
            "plausible": return_lines(os.path.join(self.record_dir, "plausible.txt")),
            "implausi":  return_lines(os.path.join(self.record_dir, "implausi.txt")),
        }
        self.messages = {}
        self.agent_resp = {}

    def level_1_repair(self, info: dict, re_patch_num=3):
        # 定位
        loc_response = self.framework["locator"].run(info, self.agent_resp)
        self.save("locator", loc_response, info['project_meta']['bug_name'])

        # 反复产 patch，直到通过测试或次数用尽
        last_fix = None
        for c in range(re_patch_num):
            logging.info(f"Generating the {c + 1}-th patch")
            fix_response = self.framework["patch"].run(info, self.agent_resp)
            self.save("patch", fix_response, info['project_meta']['bug_name'])
            last_fix = fix_response

            ok = patching_and_testing(
                patch=fix_response["aim"],
                project_meta=info["project_meta"],
                container_id=self.container_id,
            )
            if ok:
                return True, fix_response["aim"]

        return False, (last_fix["aim"] if last_fix else "")

    def save(self, role: str, response_dict: dict, bug_name: str) -> bool:
        suffix = suffix_dict[role]
        os.makedirs(os.path.join(self.response_dir, role, "aim"), exist_ok=True)
        os.makedirs(os.path.join(self.response_dir, role, "exp"), exist_ok=True)
        os.makedirs(os.path.join(self.response_dir, role, "ori"), exist_ok=True)

        if response_dict is None:
            write_line(os.path.join(self.record_dir, "failed_lst.txt"), bug_name)
            self.messages[role] = ""
            return False

        for k, v in response_dict.items():
            cur_s = "txt"
            if k == "aim":
                cur_s = suffix
                # 只把 summarizer 的结构化结果放到后续 agent 共享上下文里
                if role == "summarizer":
                    self.agent_resp[role] = json.dumps(v)
                else:
                    self.agent_resp[role] = v

            out_path = os.path.join(self.response_dir, role, k[:3], f"{bug_name}.{cur_s}")
            if cur_s == "json":
                json_pretty_dump(v, out_path)
            else:
                with open(out_path, "w", encoding="utf-8") as wf:
                    wf.write(v if isinstance(v, str) else json.dumps(v))

        self.messages[role] = response_dict.get("ori", "")
        return True

    # 简易兜底（避免示例代码报错）；请替换为你们的数据集循环
    def looping(self, **_):
        logging.warning("[Pipeline.looping] Placeholder — 请接入你们的数据集遍历逻辑。")
        return 0, 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    workpipe = Pipeline(**params)
    # 示例：这里不直接跑数据集循环，避免依赖你们的数据载入器
    # work_num, plau_num = workpipe.looping(**params)
    logging.info(f"{workpipe.hash_id} Pipeline initialized for model={params['model_name']} level={params['level']}.")
