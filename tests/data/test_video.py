from pathlib import Path
import unittest
from unittest.mock import patch

from mcwm.data.video import probe_video


class VideoProbeTest(unittest.TestCase):
    def test_missing_optional_dependency_has_clear_error(self):
        original_import = __import__

        def reject_av(name, *args, **kwargs):
            if name == "av":
                raise ImportError("forced missing av")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_av):
            with self.assertRaisesRegex(RuntimeError, "mcwm\[video\]"):
                probe_video(Path("missing.mp4"))


if __name__ == "__main__":
    unittest.main()

