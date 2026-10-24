"""Verify timed environment setup without installing packages or changing a real Conda prefix."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bootstrap


class SetupProgressTests(unittest.TestCase):
    """Keep setup output truthful and installation commands independent of progress reporting."""

    def _project(self, root, existing):
        """Build a disposable setup recipe with an optional fake existing Conda marker."""
        (root / "requirements.txt").write_text(
            "torch==2.8.0\ntorchvision==0.23.0\nnumpy==2.0.2\nPillow==11.3.0\n")
        prefix = root / ".conda"
        if existing:
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta" / "history").write_text("test environment\n")
        return prefix

    def test_verify_only_has_timed_stages_and_does_not_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            prefix = self._project(root, existing=True)
            probe = SimpleNamespace(stdout=json.dumps({"prefix": str(prefix), "kernel_test_passed": True}))
            output = io.StringIO()
            with patch("bootstrap.platform.system", return_value="Darwin"), \
                    patch("bootstrap.subprocess.check_output", side_effect=["/unrelated/base\n", "torch==2.8.0\n"]), \
                    patch("bootstrap.subprocess.run", return_value=probe) as run, redirect_stdout(output):
                executable = bootstrap.setup(root, "cpu", conda="conda", verify_only=True)
            self.assertEqual(executable, prefix / "bin" / "python")
            self.assertEqual(run.call_count, 1)
            self.assertIn("--internal-probe", run.call_args.args[0])
            self.assertIn("setup completed=5/5 (100.0%)", output.getvalue())
            self.assertIn("elapsed=", output.getvalue())
            self.assertIn("eta=unknown", output.getvalue())
            self.assertIn("Environment setup complete", output.getvalue())
            self.assertTrue((root / "verification" / "environment_cpu.json").is_file())

    def test_install_subprocesses_stay_live_and_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self._project(root, existing=False)
            error = subprocess.CalledProcessError(1, ["conda", "create"])
            output = io.StringIO()
            with patch("bootstrap.platform.system", return_value="Darwin"), \
                    patch("bootstrap.subprocess.check_output", return_value="/unrelated/base\n"), \
                    patch("bootstrap.subprocess.run", side_effect=error) as run, redirect_stdout(output):
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.setup(root, "cpu", conda="conda")
            self.assertNotIn("capture_output", run.call_args.kwargs)
            self.assertIn("setup stage=1/5 failed", output.getvalue())
            self.assertNotIn("Environment setup complete", output.getvalue())


if __name__ == "__main__":
    unittest.main()
