from pathlib import Path
import tempfile
import unittest

try:
    import av
    import numpy as np
    import torch
    from mcwm.data.video import probe_video
    from mcwm.data.visual_dataset import decode_frames_at_timestamps
except ModuleNotFoundError:
    av = None


@unittest.skipIf(av is None, "PyAV/PyTorch/NumPy is not installed")
class VisualDatasetTest(unittest.TestCase):
    def test_decode_uses_exact_pts_after_seek(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "clip.mp4"
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width = 640
                stream.height = 360
                stream.pix_fmt = "yuv420p"
                for index in range(5):
                    array = np.full((360, 640, 3), index * 40, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            timestamps = probe_video(path).frame_timestamps_ms
            decoded = decode_frames_at_timestamps(path, timestamps[1:4])
            self.assertEqual(tuple(decoded.shape), (3, 3, 360, 640))
            means = decoded.float().mean(dim=(1, 2, 3))
            self.assertTrue(torch.all(means[1:] > means[:-1]))
