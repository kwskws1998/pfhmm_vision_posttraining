"""Verify measured ETAs, resume accounting, and bounded terminal progress output."""

import io
import unittest

from pfhmm_vision.progress import Progress, format_duration


class ProgressTests(unittest.TestCase):
    """Check timing behavior with a deterministic monotonic clock."""

    def setUp(self):
        """Create one controllable clock and an isolated terminal output buffer."""
        self.now = 100.0
        self.output = io.StringIO()

    def progress(self, **kwargs):
        """Construct a stage without wall-clock sleeps or redirected global output."""
        return Progress(10, "test", clock=lambda: self.now, stream=self.output, **kwargs)

    def test_initial_eta_remains_unknown_until_work_is_measured(self):
        progress = self.progress()
        state = progress.report()
        self.assertEqual(state["completed"], 0)
        self.assertIsNone(state["remaining_seconds"])
        self.assertIn("0/10 (0.0%) elapsed=0s ETA=pending", self.output.getvalue())

    def test_elapsed_and_eta_follow_completed_work(self):
        progress = self.progress()
        self.now += 20
        state = progress.report(4)
        self.assertEqual(state["percent"], 40)
        self.assertEqual(state["elapsed_seconds"], 20)
        self.assertEqual(state["remaining_seconds"], 30)
        self.now += 10
        state = progress.report(10)
        self.assertEqual(state["remaining_seconds"], 0)
        self.assertIn("10/10 (100.0%) elapsed=30s ETA=0s", self.output.getvalue())

    def test_resume_preserves_historical_time_and_excludes_time_between_invocations(self):
        progress = self.progress(completed=4, elapsed_seconds=20)
        state = progress.report()
        self.assertEqual(state["elapsed_seconds"], 20)
        self.assertEqual(state["remaining_seconds"], 30)
        self.now += 5
        state = progress.snapshot(5)
        self.assertEqual(state["elapsed_seconds"], 25)
        self.assertEqual(state["remaining_seconds"], 25)

    def test_interval_throttles_output_but_completion_and_force_are_immediate(self):
        progress = self.progress(min_interval=15)
        progress.report()
        self.now += 1
        progress.report(1)
        self.assertEqual(len(self.output.getvalue().splitlines()), 1)
        progress.report(2, detail="checkpoint", force=True)
        self.assertEqual(len(self.output.getvalue().splitlines()), 2)
        self.now += 16
        progress.report(3)
        self.assertEqual(len(self.output.getvalue().splitlines()), 3)
        progress.report(10)
        self.assertEqual(len(self.output.getvalue().splitlines()), 4)

    def test_invalid_or_decreasing_progress_is_rejected(self):
        progress = self.progress(completed=5)
        for completed in (-1, 4, 11, 5.5, True):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                progress.snapshot(completed)

    def test_invalid_initial_work_and_timing_are_rejected(self):
        for kwargs in ({"total": 0}, {"total": True}, {"completed": 11},
                       {"completed": -1}, {"elapsed_seconds": -1},
                       {"elapsed_seconds": float("inf")}, {"min_interval": -1},
                       {"min_interval": float("nan")}):
            settings = {"total": 10, "description": "test", **kwargs}
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Progress(**settings)

    def test_duration_formatting_handles_long_runs(self):
        for seconds, expected in ((None, "pending"), (0, "0s"), (65, "1m 05s"),
                                  (3601, "1h 00m 01s"), (90061, "25h 01m 01s")):
            with self.subTest(seconds=seconds):
                self.assertEqual(format_duration(seconds), expected)
        for seconds in (-1, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                format_duration(seconds)


if __name__ == "__main__":
    unittest.main()
