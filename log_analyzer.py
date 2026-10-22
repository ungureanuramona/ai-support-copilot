
import re
from collections import Counter
from datetime import datetime


ERROR_PATTERNS = {
    "Database connection issue": [
        r"database connection",
        r"connection pool exhausted",
        r"sql.*exception",
    ],
    "Network timeout": [
        r"connection timed out",
        r"read timeout",
        r"socket timeout",
    ],
    "HTTP service failure": [
        r"http 5\d\d",
        r"service unavailable",
        r"bad gateway",
    ],
    "Authentication failure": [
        r"authentication failed",
        r"invalid credentials",
        r"unauthorized",
    ],
}


def analyze_logs(log_text: str) -> dict:
    if not log_text.strip():
        return {
            "total_lines": 0,
            "errors": 0,
            "warnings": 0,
            "detected_issues": {},
            "events": [],
        }

    lines = log_text.splitlines()
    events = []
    detected_issues = Counter()
    error_count = 0
    warning_count = 0

    for line_number, line in enumerate(lines, start=1):
        upper_line = line.upper()

        if re.search(r"\b(ERROR|FATAL|CRITICAL)\b", upper_line):
            severity = "ERROR"
            error_count += 1
        elif re.search(r"\b(WARN|WARNING)\b", upper_line):
            severity = "WARNING"
            warning_count += 1
        else:
            continue

        matched_issues = []

        for issue_name, patterns in ERROR_PATTERNS.items():
            if any(
                re.search(pattern, line, re.IGNORECASE)
                for pattern in patterns
            ):
                detected_issues[issue_name] += 1
                matched_issues.append(issue_name)

        timestamp_match = re.search(
            r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
            line,
        )

        events.append({
            "line_number": line_number,
            "timestamp": (
                timestamp_match.group(0)
                if timestamp_match
                else None
            ),
            "severity": severity,
            "message": line.strip(),
            "issues": matched_issues,
        })

    return {
        "total_lines": len(lines),
        "errors": error_count,
        "warnings": warning_count,
        "detected_issues": dict(detected_issues),
        "events": events,
    }


if __name__ == "__main__":
    sample_logs = """
2026-10-22 10:15:02 ERROR Database connection timeout
2026-10-22 10:15:03 WARN Connection pool exhausted
2026-10-22 10:15:04 ERROR HTTP 503 Service Unavailable
"""

    from pprint import pprint

    pprint(analyze_logs(sample_logs))
