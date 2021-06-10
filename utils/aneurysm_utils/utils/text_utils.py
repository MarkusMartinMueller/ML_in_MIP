import re

_SIMPLIFY_STRING_PATTERN = re.compile(r"[^a-zA-Z0-9-]")


def safe_str(obj) -> str:
    try:
        return str(obj)
    except UnicodeEncodeError:
        return obj.encode("ascii", "ignore").decode("ascii")


def simplify(text) -> str:
    text = safe_str(text)
    return _SIMPLIFY_STRING_PATTERN.sub("-", clean_whitespaces(text.strip())).lower()


def clean_whitespaces(text: str) -> str:
    return " ".join(text.split())


def simplify_duration(duration_ms: int) -> str:
    if not duration_ms:
        return ""

    from dateutil.relativedelta import relativedelta as rd

    intervals = ["days", "hours", "minutes", "seconds"]
    rel_date = rd(microseconds=duration_ms * 1000)
    return " ".join(
        "{} {}".format(getattr(rel_date, k), k)
        for k in intervals
        if getattr(rel_date, k)
    )
