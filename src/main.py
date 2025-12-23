# main.py
import logging
import queue
import threading
import time

from reader import RTSPReader
from tt_processor import TableTennisProcessor
from writer import RTSPWriter

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")

# Настройки
import os

WIDTH = int(os.getenv("WIDTH", "1920"))
HEIGHT = int(os.getenv("HEIGHT", "1080"))
FPS = int(os.getenv("FPS", "30"))

INPUT_QUEUE_SIZE = int(os.getenv("INPUT_QUEUE_SIZE", "60"))
OUTPUT_QUEUE_SIZE = int(os.getenv("OUTPUT_QUEUE_SIZE", "60"))

input_queue = queue.Queue(maxsize=INPUT_QUEUE_SIZE)
output_queue = queue.Queue(maxsize=OUTPUT_QUEUE_SIZE)

# RTSP URL
INPUT_URL = os.getenv("INPUT_URL", "rtsp://147.45.159.99:8554/live/tennis")
OUTPUT_URL = os.getenv("OUTPUT_URL", "rtsp://147.45.159.99:8554/live/processed_tennis")

# Потоки чтения и записи
reader = RTSPReader(INPUT_URL, WIDTH, HEIGHT, FPS, input_queue)
writer = RTSPWriter(output_queue, OUTPUT_URL, WIDTH, HEIGHT, FPS)

reader.start()
writer.start()

# Инициализация YOLO + логика игры
processor = TableTennisProcessor(
    model_path=os.getenv("MODEL_PATH", "../model/ppv_yolo11s_based.pt"),
    corners_json=os.getenv("CORNERS_JSON", "table_corners4.json"),
)


def processing_loop():
    while True:
        try:
            frame = input_queue.get(timeout=0.03)
        except queue.Empty:
            continue

        try:
            processed_frame, _ = processor.process_frame(frame)
        except Exception as e:
            log.error(f"Processing error: {e}, forwarding raw frame")
            processed_frame = frame.copy()

        try:
            output_queue.put_nowait(processed_frame)
        except queue.Full:
            log.warning("Output queue full — dropping frame")


if __name__ == "__main__":
    log.info("Starting pipeline threads")
    processing_thread = threading.Thread(target=processing_loop, daemon=True)
    processing_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopping pipeline")
        input_queue.put(None)
        output_queue.put(None)
        reader.join()
        writer.join()
