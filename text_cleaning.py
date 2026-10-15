import re


def clean_text(text):
    # Collapse line breaks and repeated whitespace only. An earlier version also
    # removed single spaces between lowercase letters, which glued words together
    # ("theuseofbiometric...") and cut retrieval hit@4 from 72% to 44% (see
    # evaluate_retrieval.py).
    return re.sub(r'\s+', ' ', text).strip()
