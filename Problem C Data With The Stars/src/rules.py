"""Rule definitions and scoring logic."""

import re

# "Eliminated Week k" -> extract k
ELIMINATED_WEEK_PATTERN = re.compile(r"Eliminated Week\s+(\d+)", re.IGNORECASE)

# "Withdrew" is not an elimination week string; handle separately
WITHDREW_STR = "Withdrew"


def parse_elimination_week(results: str) -> int | None:
    """Parse elimination week from results string. Returns None if not eliminated (e.g. Place, Withdrew)."""
    if not results or results.strip() == "":
        return None
    m = ELIMINATED_WEEK_PATTERN.search(results.strip())
    return int(m.group(1)) if m else None


def is_withdrew(results: str) -> bool:
    """True if results indicate withdrawal."""
    return results is not None and WITHDREW_STR.lower() in (results or "").lower()
