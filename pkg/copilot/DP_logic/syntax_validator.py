import re
from typing import Callable
import logging

logger = logging.getLogger(__name__)

AGGREGATION_OPS = [
    'sum', 'min', 'max', 'avg', 'count',
    'stddev', 'stdvar', 'count_values',
    'bottomk', 'topk', 'quantile',
]

OVER_TIME_FUNCS = [
    'avg_over_time', 'sum_over_time', 'min_over_time', 'max_over_time',
    'count_over_time', 'stddev_over_time', 'stdvar_over_time',
    'last_over_time', 'present_over_time', 'quantile_over_time',
]


def check_balanced_braces(promql: str) -> list[str]:
    issues = []
    if promql.count('{') != promql.count('}'):
        issues.append("unbalanced curly braces")
    if promql.count('[') != promql.count(']'):
        issues.append("unbalanced square brackets")
    if promql.count('(') != promql.count(')'):
        issues.append("unbalanced parentheses")
    return issues


def check_no_conflicting_modifiers(promql: str) -> list[str]:
    """Catches: sum(...) by (...) without (...) — can't use both."""
    issues = []
    if re.search(r'\b(by|without)\s*\([^)]*\)\s*(by|without)\s*\(', promql, re.IGNORECASE):
        issues.append("cannot use both 'by' and 'without' modifiers together")
    return issues


def check_valid_aggregations(promql: str) -> list[str]:
    """Catches: sum over (...) — wrong keyword, should be 'by' or 'without'."""
    issues = []
    for op in AGGREGATION_OPS:
        if re.search(rf'\b{op}\s+over\s*\(', promql, re.IGNORECASE):
            issues.append(f"'{op} over' detected — use '{op} by' or '{op} without' instead")
            break
    return issues


def check_range_vector_in_aggregation(promql: str) -> list[str]:
    """
    Catches: sum(metric[5m]) — range vector used directly in aggregation.
    Should be: sum(rate(metric[5m])) or sum(avg_over_time(metric[5m])).
    """
    issues = []
    pattern = (
        r'\b(' + '|'.join(AGGREGATION_OPS) + r')'
        r'\s*(?:by|without)?\s*\([^)]*\[[0-9]+[smhdwy]\]\s*\)'
    )
    if re.search(pattern, promql, re.IGNORECASE):
        issues.append(
            "range vector used directly in aggregation — "
            "wrap with rate(), avg_over_time(), or similar function first"
        )
    return issues


def check_over_time_has_range(promql: str) -> list[str]:
    """
    Catches: avg_over_time(metric) — missing required range vector.
    Should be: avg_over_time(metric[5m]).
    """
    issues = []
    for func in OVER_TIME_FUNCS:
        if re.search(rf'\b{func}\s*\([^)]*\)', promql, re.IGNORECASE):
            if not re.search(rf'\b{func}\s*\([^)]*\[[0-9]+[smhdwy]\]', promql, re.IGNORECASE):
                issues.append(f"'{func}' requires a range vector (e.g., metric[5m]) as argument")
                break
    return issues


# All rules run by validate()
_rules = [
    {"name": "balanced_braces",            "check": check_balanced_braces},
    {"name": "no_conflicting_modifiers",   "check": check_no_conflicting_modifiers},
    {"name": "valid_aggregations",         "check": check_valid_aggregations},
    {"name": "range_vector_in_aggregation","check": check_range_vector_in_aggregation},
    {"name": "over_time_has_range",        "check": check_over_time_has_range},
]


def validate_promql(promql: str) -> list[str]:
    """
    Run all syntax checks against a PromQL string.
    Returns a list of issue strings (empty = no issues found).
    """
    all_issues = []
    for rule in _rules:
        fn: Callable[[str], list[str]] = rule["check"]  # type: ignore[assignment]
        all_issues.extend(fn(promql))
    if all_issues:
        logger.warning(f"Syntax issues in PromQL: {', '.join(all_issues)}")
    return all_issues
