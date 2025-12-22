from ultralytics.models import YOLO
import cv2
import numpy as np

# Путь к вашей кастомной модели (с классами 0, 1, 2)
model = YOLO("model/ppv_yolo11s_based.pt")  # замените на реальный путь, например "best.pt"

# Видео
video_path = "videos/test_1.mp4"  # замените
cap = cv2.VideoCapture(video_path)

# Параметры видео для вывода
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output3_tracking_only_ball.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Словарь для хранения траекторий: track_id -> список центров (x, y)
trajectories = {}

# Цвета (BGR)
BALL_COLOR = (0, 255, 0)      # зелёный для мяча
OTHER_COLOR = (255, 0, 0)    # синий для остальных классов
TRAJECTORY_COLOR = (0, 255, 255)  # жёлтый для линии траектории
TRAJECTORY_THICKNESS = 3
MAX_TRAJECTORY_POINTS = 50  # сколько последних точек хранить

trajectories = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ТОЛЬКО ДЕТЕКЦИЯ (без track!) — здесь модель будет детектировать мяч как раньше
    results = model(
        source=frame,
        conf=0.25,          # понизьте, если нужно больше детекций
        iou=0.7,            # NMS IoU
        verbose=False
    )[0]

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            if cls == 0:  # только мяч — рисуем траекторию (простой "трейл" без ID)
                color = BALL_COLOR
                label = f"Ball {conf:.2f}"

                # Простая траектория: храним последние точки центра
                track_id = 0  # поскольку обычно один мяч, фиксированный ID
                if track_id not in trajectories:
                    trajectories[track_id] = []
                trajectories[track_id].append((center_x, center_y))
                if len(trajectories[track_id]) > MAX_TRAJECTORY_POINTS:
                    trajectories[track_id].pop(0)

                pts = np.array(trajectories[track_id], np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], isClosed=False, color=TRAJECTORY_COLOR, thickness=TRAJECTORY_THICKNESS)

            else:
                color = OTHER_COLOR
                label = f"Class {cls} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    # Опционально: показ в реальном времени
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Готово! Видео сохранено как output_tracking_only_ball.mp4")