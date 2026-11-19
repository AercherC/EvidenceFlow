"""证据摘录的连续原文匹配，供决策评估与生成合同复用。"""

import re
import unicodedata


def quote_in_source(quote: str, source: str) -> bool:
    """容许 PDF 排版差异，同时保留词边界、普通连字符和数值符号。"""
    quote, source = (unicodedata.normalize("NFKC", value) for value in (quote, source))
    compact = " ".join(quote.split())
    if not compact:
        return False
    # 不从更长词或数值中截取，尤其不能把 -0.75 的尾部当成 +0.75。
    prefix = r"(?<![\w.+\-−])" if compact[0].isdigit() else r"(?<!\w)" if compact[0].isalnum() else ""
    suffix = r"(?!\w|\.\d)" if compact[-1].isdigit() else r"(?!\w)" if compact[-1].isalnum() else ""
    if re.search(prefix + re.escape(compact) + suffix, " ".join(source.split())):
        return True
    # 只在真实行尾允许省掉连字符。正文中的 x-y 不能被匹配为 xy。
    quote = re.sub(r"(?<=[^\W\d_])-[ \t]*\r?\n\s*(?=[^\W\d_])", "", quote)
    compact = " ".join(quote.split())
    pattern = []
    for index, character in enumerate(compact):
        if character == " " and index and index + 1 < len(compact) and compact[index - 1].isalpha() and compact[index + 1].isalpha():
            pattern.append(r"(?:\s+|-[ \t]*\r?\n\s*)")
        else:
            pattern.append(r"\s+" if character == " " else r"-\s*" if character == "-" else re.escape(character))
        if character.isalpha() and index + 1 < len(compact) and compact[index + 1].isalpha():
            pattern.append(r"(?:-[ \t]*\r?\n\s*)?")
    return re.search(prefix + "".join(pattern) + suffix, source) is not None
