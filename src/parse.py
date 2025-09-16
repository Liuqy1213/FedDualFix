import re
import logging
from typing import List, Optional, Tuple


class NoCodeError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def parse_code(text: str) -> List[str]:
    """
    提取代码块：优先 ```...```；其次 === 分隔；最后行内 `
    返回匹配到的代码块列表；若一个也没有则抛 NoCodeError。
    """
    patterns = [
        r'```(?:[^\n]*\n)?(.*?)```',
        r'```(?:[^\n]*\n)?(.*?)===',
        r'^(.*?)```',
        r'`(?:[^\n]*\n)?(.*?)`',
        r'```(?:[^\n]*\n)?(.*?)$',
    ]
    for pat in patterns:
        tmp = re.findall(pat, text, re.DOTALL)
        if tmp and tmp[0].strip():
            return tmp
    raise NoCodeError(f"Cannot extract any code from:\n@@@@@\n{text}\n@@@@@\n")


def parse_exp(text: str) -> str:
    """
    提取解释块：优先 === ... ===；否则到结尾；否则到 ``` 之前。
    """
    patterns = [
        r'===(?:[^\n]*\n)?(.*?)===',
        r'===(?:[^\n]*\n)?(.*?)$',
        r'^(?:[^\n]*\n)?(.*?)```',
    ]
    for pat in patterns:
        tmp = re.findall(pat, text, re.DOTALL)
        if tmp:
            return "\n".join(tmp)
    logging.warning("This response doesn't explain the repairing")
    return ""


def remove_comment(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//.*', '', code)
    return re.sub(r'^\s*$', '', code, flags=re.MULTILINE)  # 移除空行


def remove_whitespace(line: str) -> str:
    return line.replace('\n', '').replace(' ', '')


def two_lines_match(line1: Optional[str], line2: Optional[str]) -> bool:
    """
    忽略注释/空白后的行匹配。两行都需要非空。
    """
    if not line1 or not line2:
        return False
    s1, s2 = line1.strip(), line2.strip()
    if not s1 or not s2:
        return False
    # 注释行对注释行：完全比较
    if s1.startswith("//") and s2.startswith("//"):
        return bool(remove_whitespace(s1) and remove_whitespace(s2) and remove_whitespace(s1) == remove_whitespace(s2))
    # 否则去掉行尾注释后再比
    l1 = remove_whitespace(line1.split("//")[0])
    l2 = remove_whitespace(line2.split("//")[0])
    return bool(l1 and l2 and l1 == l2)


def exist_line(line: str, mylst: Optional[List[str]]) -> bool:
    if mylst is None:
        return True
    for l in mylst:
        if two_lines_match(l, line):
            return True
    return False


def is_valid_line(line: str, length: int = 0) -> bool:
    """
    有效行：非空白、不是编辑标记(+/missing/buggy)开头、长度>length。
    """
    if line is None:
        return False
    nowh = remove_whitespace(line)
    if len(nowh) <= length:
        return False
    s = line.strip()
    if not s:
        return False
    if s[0] == "+":
        return False
    if "missing" in line or "buggy" in line:
        return False
    return True


def search_valid_line(lines: List[str], start_idx: int, mode: str,
                      degree: int = 1, existing: Optional[List[str]] = None) -> Optional[Tuple[int, str]]:
    """
    从 start_idx 向前(pre)/向后(post)找第 degree 个有效行（存在于 existing 列表）。
    """
    incre = -1 if mode == "pre" else 1
    cur_idx = start_idx + incre
    while 0 <= cur_idx < len(lines):
        if is_valid_line(lines[cur_idx]) and exist_line(lines[cur_idx], existing):
            degree -= 1
            if degree == 0:
                return (cur_idx, lines[cur_idx])
        cur_idx += incre
    return None


def matching_with_comments(aim_line: str, matched: List[int], code_lines: List[str]) -> List[int]:
    """ 完全匹配（含注释/空格归一） """
    return [m for m in matched if remove_whitespace(aim_line) == remove_whitespace(code_lines[m])]


def matching_lines(aim_line: Optional[str], code_lines: List[str], stop_at_first_match: bool = False) -> List[int]:
    if aim_line is None:
        return []
    out = []
    for idx, cl in enumerate(code_lines):
        if two_lines_match(aim_line, cl):
            out.append(idx)
            if stop_at_first_match:
                return [idx]
    return out


def matching_neighbor(aim_codes: List[str], aim_idx: int, raw_codes: List[str],
                      matched: List[int], existing: bool = False, degree_limit: int = 5) -> List[int]:
    """
    通过邻接行 disambiguation：向前/向后逐级寻找匹配，尝试唯一化。
    """
    existing_pool = raw_codes if existing else None
    pre_now, post_now = list(matched), list(matched)
    pre_also, post_also = [], []

    for degree in range(1, degree_limit + 1):
        aim_pre = search_valid_line(aim_codes, aim_idx, "pre", degree=degree, existing=existing_pool)
        aim_post = search_valid_line(aim_codes, aim_idx, "post", degree=degree, existing=existing_pool)

        if aim_pre is not None:
            for mi in pre_now:
                pre_match = search_valid_line(raw_codes, mi, "pre", degree=degree)
                if (pre_match is not None) and two_lines_match(aim_pre[1], pre_match[1]):
                    pre_also.append(mi)
            if len(pre_also) == 1:
                return pre_also

        if aim_post is not None:
            for mi in post_now:
                post_match = search_valid_line(raw_codes, mi, "post", degree=degree)
                if (post_match is not None) and two_lines_match(aim_post[1], post_match[1]):
                    post_also.append(mi)
            if len(post_also) == 1:
                return post_also

        inter = set(pre_also) & set(post_also)
        if pre_also and post_also and len(inter) == 1:
            return list(inter)
        if not pre_also and not post_also:
            return []

        pre_now, post_now = list(pre_also), list(post_also)
        pre_also, post_also = [], []

    return []


def unique_matching(resp_lines: List[str], code_lines: List[str], resp_cur_idx: int,
                    resp_cur_line: Optional[str] = None, existing: bool = False) -> int:
    """
    将响应中的某一行在原代码中唯一定位；返回唯一匹配的下标；
    若 0 个匹配 → -2；若多匹配但未唯一化 → -1。
    """
    target = resp_lines[resp_cur_idx] if resp_cur_line is None else resp_cur_line
    matched = matching_lines(target, code_lines)
    if len(matched) == 1:
        return matched[0]
    if len(matched) == 0:
        return -2

    neighbor = matching_neighbor(resp_lines, resp_cur_idx, code_lines, matched, degree_limit=5, existing=existing)
    if len(neighbor) == 1:
        return neighbor[0]
    logging.debug(f"unique_matching failed for '{target}' with {len(matched)} candidates")
    return -1
