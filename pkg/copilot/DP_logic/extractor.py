import re
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Extraction patterns — tried in order, first match wins
extraction_patterns = [
    {"name": "code_block_without_newlines", "regex": r"```(?:\w+)?\s*(.*?)\s*```"},
    {"name": "query_label", "regex": r"query:\s*([^\n]+)"},
    {
        "name": "common_promql_functions",
        "regex": (
            r"((?:sum|avg|min|max|count|rate|increase|histogram_quantile|"
            r"avg_over_time|sum_over_time|max_over_time|min_over_time|"
            r"count_over_time|stddev_over_time|irate|delta|deriv|predict_linear)"
            r"\s*\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\)"
            r"(?:\s*(?:by|without)\s*\([^)]*\))?)"
        ),
    },
]

# Cleanup transforms — applied sequentially to the extracted query
cleanup_transforms: list[dict[str, str | Callable[[str], str]]] = [
    {"name": "remove_language_identifiers", "fn": lambda p: re.sub(r'^(promql|markdown|sql|python|bash|plaintext)\s*\n', '', p, flags=re.IGNORECASE)},
    {"name": "remove_comments", "fn": lambda p: re.sub(r'#.*', '', p)},
    {"name": "remove_leading_brackets", "fn": lambda p: re.sub(r'^\s*\[\s*', '', p)},
    {"name": "remove_trailing_brackets", "fn": lambda p: re.sub(r'\s*\]\s*$', '', p)},
    {"name": "strip_whitespace", "fn": lambda p: p.strip()},
]


def extract_promql_from_response(response_text: str) -> str:
    """
    Extracts a PromQL query from an LLM response by trying multiple
    extraction patterns in order. The first match is cleaned up and returned.
    """
    extracted = None
    matched_pattern = None

    for pattern in extraction_patterns:
        match = re.search(pattern["regex"], response_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            matched_pattern = pattern["name"]
            logger.info(f"Matched extraction pattern: '{matched_pattern}'")
            break

    if not extracted:
        logger.error(
            f"No PromQL query found in response. Full response:\n{response_text}"
        )
        raise ValueError(
            f"No valid PromQL found in response. Response length: {len(response_text)} chars"
        )

    # Apply cleanup transforms sequentially
    for transform in cleanup_transforms:
        fn: Callable[[str], str] = transform["fn"]  # type: ignore[assignment]
        extracted = fn(extracted)

    if not extracted:
        logger.error("Extracted PromQL is empty after cleanup")
        raise ValueError("Empty PromQL query after cleanup")

    logger.info(f"Extracted PromQL: {extracted}")
    return extracted