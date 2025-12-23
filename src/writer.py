# writer.py
import subprocess
import threading
import queue
import logging

log = logging.getLogger("writer")

class RTSPWriter(threading.Thread):
    def __init__(self, input_queue, output_url, width, height, fps):
        super().__init__()
        self.input_queue = input_queue
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.proc = None
        self.daemon = True

    def run(self):
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "-fflags", "nobuffer",
            self.output_url
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        last_frame = None

        while True:
            try:
                frame = self.input_queue.get(timeout=0.03)
                if frame is None:
                    break
                last_frame = frame
            except queue.Empty:
                if last_frame is None:
                    continue
                frame = last_frame

            try:
                self.proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                log.error("Broken pipe — FFmpeg writer stopped")
                break

        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
