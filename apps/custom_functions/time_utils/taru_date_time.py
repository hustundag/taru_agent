# custom_functions/time_utils/taru_date_time.py

import datetime
import calendar
from TaruAgent import tool

# Helper routines for month/year arithmetic
def _add_months(dt: datetime.datetime, months: int) -> datetime.datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)

def _add_years(dt: datetime.datetime, years: int) -> datetime.datetime:
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Handle leap
        return dt.replace(year=dt.year + years, day=28)

# Map unit names to the appropriate adjustment function
SUPPORTED_UNITS: dict[str, callable] = {
    "seconds": lambda dt, val, direction: dt + datetime.timedelta(
        seconds= val if direction == "after" else -val
    ),
    "minutes": lambda dt, val, direction: dt + datetime.timedelta(
        minutes= val if direction == "after" else -val
    ),
    "hour":    lambda dt, val, direction: dt + datetime.timedelta(
        hours=   val if direction == "after" else -val
    ),
    "day":     lambda dt, val, direction: dt + datetime.timedelta(
        days=    val if direction == "after" else -val
    ),
    "week":    lambda dt, val, direction: dt + datetime.timedelta(
        weeks=   val if direction == "after" else -val
    ),
    "month":   lambda dt, val, direction: _add_months(
        dt, val if direction == "after" else -val
    ),
    "year":    lambda dt, val, direction: _add_years(
        dt, val if direction == "after" else -val
    ),
}

@tool(
    name="relative_time",
    description=(
        "- unit: The unit of time measurement. Allowed values are seconds, minutes, hour, day, week, month, year.\n"
        "- direction: The direction of time adjustment. Allowed values are before, after.\n"
        "- value: Integer value for the amount of time to adjust. Zero means current date & time.\n\n"
        "Returns a dict with:\n"
        "  - iso8601:   ISO 8601 timestamp\n"
        "  - rfc2822:  RFC 2822 timestamp\n"
        "  - rfc33339:  RFC 3339 timestamp (same as ISO 8601 with timezone)\n"
        "  - unix_timestamp: Unix epoch seconds\n"
        "  - readable: humanreadable '%Y-%m-%d %H:%M:%S %Z'"
    ),
)
def relative_time(unit: str, direction: str, value: int) -> dict:
    """
    Calculate a time offset from now (UTC). See description for usage.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    if unit not in SUPPORTED_UNITS:
        return {
            "error": (
                f"Unsupported unit: {unit}. "
                "Supported: seconds, minutes, hour, day, week, month, year."
            )
        }

    if direction not in ("before", "after"):
        return {
            "error": (
                f"Unsupported direction: {direction}. "
                "Must be either 'before' or 'after'."
            )
        }

    if not isinstance(value, int):
        return {"error": "Value must be an integer."}

    adjusted = SUPPORTED_UNITS[unit](now, value, direction)

    return {
        "result": {
            "iso8601":        adjusted.isoformat(),
            "rfc2822":        adjusted.strftime("%a, %d %b %Y %H:%M:%S %z"),
            "rfc3339":        adjusted.isoformat(),
            "unix_timestamp": int(adjusted.timestamp()),
            "readable":       adjusted.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
    }
