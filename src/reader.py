# reader.py
import subprocess
import threading
import queue
import logging
import numpy as np

log = logging.getLogger("reader")

class RTSPReader(threading.Thread):
    def __init__(self, url, width, height, fps, output_queue, queue_size=60):
        super().__init__()
        self.url = url
        self.width = width
        self.height = height
        self.fps = fps
        self.output_queue = output_queue
        self.queue_size = queue_size
        self.proc = None
        self.daemon = True

    def run(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_size = self.width * self.height * 3
        while True:
            raw_frame = self.proc.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                log.warning("EOF or broken frame")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((self.height, self.width, 3))
            try:
                self.output_queue.put_nowait(frame)
            except queue.Full:
                log.warning("Input queue full — dropping frame")
