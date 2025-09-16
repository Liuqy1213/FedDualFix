import logging
from typing import Any, Optional, Tuple

# 尝试懒加载依赖（javalang / APTED）
try:
    import javalang
    from apted import APTED  # Config 可选，不强依赖
    _HAS_JAVA = True
except Exception as e:
    logging.warning(f"[myast] AST deps not available: {e}")
    _HAS_JAVA = False

AstTuple = Optional[Tuple[str, list]]


def code2ast(code: str, language: str = "java") -> Optional[Any]:
    """
    Parse source code to AST. Currently supports Java via javalang.
    Returns None if deps missing or language unsupported / parse failed.
    """
    if language.lower() != "java" or not _HAS_JAVA:
        return None
    try:
        return javalang.parse.parse(code)
    except Exception as e:
        logging.warning(f"[myast] parse failed: {e}")
        return None


def ast_to_tuple(node: Any) -> AstTuple:
    """
    Convert AST to a (TypeName, [children...]) tuple that APTED 接口可接受。
    None-safe：传入非 javalang 节点或 None → 返回 None。
    """
    if not _HAS_JAVA or node is None:
        return None
    if isinstance(node, javalang.ast.Node):
        children = []
        try:
            for _, child in node.children():
                t = ast_to_tuple(child)
                if t is not None:
                    children.append(t)
        except Exception:
            pass
        return (node.__class__.__name__, children)
    return None


def ast_dis(tree1: Any, tree2: Any) -> Optional[int]:
    """
    Compute tree edit distance via APTED on tupleized ASTs.
    任一为空 → 返回 None 表示无法度量。
    """
    if not _HAS_JAVA:
        return None
    t1, t2 = ast_to_tuple(tree1), ast_to_tuple(tree2)
    if t1 is None or t2 is None:
        return None
    try:
        apted = APTED(t1, t2)
        return apted.compute()
    except Exception as e:
        logging.warning(f"[myast] APTED compute failed: {e}")
        return None








