"""Low-overhead terminal progress with cumulative elapsed time and an observed-rate ETA."""

import math
import sys
import time


def format_duration(seconds):
    """Render nonnegative durations compactly and leave an unmeasured ETA explicit."""
    if seconds is None:
        return "pending"
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Duration must be finite and nonnegative")
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class Progress:
    """Report completed work, percentage, cumulative elapsed time, and an empirical ETA."""

    def __init__(self, total, description, completed=0, elapsed_seconds=0.0,
                 min_interval=15.0, clock=time.monotonic, stream=None):
        """Initialize a stage, optionally restoring committed work and its measured time."""
        if type(total) is not int or total <= 0:
            raise ValueError("Progress total must be a positive integer")
        if type(completed) is not int or not 0 <= completed <= total:
            raise ValueError("Completed work must be an integer between zero and total")
        if (not math.isfinite(elapsed_seconds) or elapsed_seconds < 0 or
                not math.isfinite(min_interval) or min_interval < 0):
            raise ValueError("Elapsed time and report interval must be finite and nonnegative")
        self.total = total
        self.description = description
        self.completed = completed
        self.elapsed_seconds = elapsed_seconds
        self.min_interval = min_interval
        self.clock = clock
        self.stream = stream
        self.started = clock()
        self.last_report = None

    def _snapshot(self, completed, now):
        """Validate monotone progress and estimate remaining work at the measured average rate."""
        if completed is not None:
            if type(completed) is not int or not self.completed <= completed <= self.total:
                raise ValueError("Completed work must increase monotonically up to total")
            self.completed = completed
        elapsed = self.elapsed_seconds + max(0.0, now - self.started)
        remaining = (elapsed * (self.total - self.completed) / self.completed
                     if self.completed else None)
        return {"completed": self.completed, "total": self.total,
                "percent": 100.0 * self.completed / self.total,
                "elapsed_seconds": elapsed, "remaining_seconds": remaining}

    def snapshot(self, completed=None):
        """Return JSON-compatible progress values without printing a status line."""
        return self._snapshot(completed, self.clock())

    def report(self, completed=None, detail="", force=False):
        """Print a flushed status line at the configured interval or when a stage completes."""
        now = self.clock()
        status = self._snapshot(completed, now)
        if (force or self.last_report is None or self.completed == self.total or
                now - self.last_report >= self.min_interval):
            message = (f"{self.description} {self.completed}/{self.total} "
                       f"({status['percent']:.1f}%) "
                       f"elapsed={format_duration(status['elapsed_seconds'])} "
                       f"ETA={format_duration(status['remaining_seconds'])}")
            if detail:
                message += f" {detail}"
            print(message, file=self.stream if self.stream is not None else sys.stdout, flush=True)
            self.last_report = now
        return status
