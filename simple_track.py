import json
import sys

from ultralytics.models import YOLO
import cv2
import numpy as np
import argparse
from game_logic import *

def load_table_corners(json_path):
    """
    Загружает координаты углов стола из JSON файла
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Используем только исходные углы, а не отсортированные
    corners_data = data["corners"]
    corners = np.array([[corner["x"], corner["y"]] for corner in corners_data], dtype=np.float32)

    return corners


def compute_homography_matrix(src_points):
    """
    Вычисляет матрицу гомографии для перспективной трансформации
    """
    # Целевые точки для прямоугольника (стол для настольного тенниса 274x152.5 см)
    # Уменьшим масштаб для отображения: 2740x1525 пикселей
    dst_points = np.array(
        [
            [0, 0],  # верх-лево
            [2740, 0],  # верх-право
            [2740, 1525],  # низ-право
            [0, 1525],  # низ-лево
        ],
        dtype=np.float32,
    )

    # Вычисляем матрицу гомографии
    homography_matrix, mask = cv2.findHomography(src_points, dst_points)
    return homography_matrix, dst_points

def get_zone(x, y):
    if x < MID_X and y < MID_Y:
        return 3  # низ-лево
    elif x >= MID_X and y < MID_Y:
        return 1  # верх-лево
    elif x < MID_X and y >= MID_Y:
        return 4  # низ-право
    else:
        return 2  # верх-право


# Путь к вашей кастомной модели (с классами 0, 1, 2)
model = YOLO("model/ppv_yolo11s_based.pt")  # замените на реальный путь, например "best.pt"

# Видео
video_path = "videos/test_4.mp4"  # замените
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
TABLE_W = 2740
TABLE_H = 1525

MID_X = TABLE_W // 2
MID_Y = TABLE_H // 2

trajectories = {}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--corners", help="JSON file with table corners", default="table_corners4.json"
)
args = parser.parse_args()

corners_path = args.corners

# Загружаем координаты углов стола
src_corners = load_table_corners(corners_path)
print(f"Загружены углы стола: {src_corners}")

# Вычисляем матрицу гомографии
homography_matrix, dst_points = compute_homography_matrix(src_corners)
print("Матрица гомографии вычислена")

if homography_matrix is None:
    print("ERROR: Could not compute homography matrix")
    sys.exit(0)

top_view_trajectories = {}
MAX_TOP_VIEW_POINTS = 50

# История мяча
ball_history = []

MAX_HISTORY = 10

# Флаги
last_event = None
event_cooldown = 0
EVENT_COOLDOWN = 8

ball_state = BallState()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Вид сверху (размер соответствует dst_points)
    top_view = np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8)
    # Вертикальная линия
    cv2.line(top_view, (MID_X, 0), (MID_X, TABLE_H), (255, 255, 255), 2)

    # Горизонтальная линия
    cv2.line(top_view, (0, MID_Y), (TABLE_W, MID_Y), (255, 255, 255), 2)

    # Подписи зон
    cv2.putText(top_view, "Zone 3", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(top_view, "Zone 1", (MID_X + 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(top_view, "Zone 4", (100, MID_Y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(top_view, "Zone 2", (MID_X + 100, MID_Y + 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Контур стола на виде сверху
    table_outline = np.array(dst_points, dtype=np.int32)
    cv2.polylines(top_view, [table_outline], True, (0, 255, 255), 3)

    # ТОЛЬКО ДЕТЕКЦИЯ (без track!) — здесь модель будет детектировать мяч как раньше
    results = model(
        source=frame,
        conf=0.25,          # понизьте, если нужно больше детекций
        iou=0.7,            # NMS IoU
        verbose=False
    )[0]

    # Рисуем углы стола на исходном кадре
    for i, corner in enumerate(src_corners):
        cv2.circle(frame, (int(corner[0]), int(corner[1])), 10, (0, 255, 255), -1)
        cv2.putText(
            frame,
            str(i + 1),
            (int(corner[0]) + 10, int(corner[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    # Рисуем контур стола
    pts = src_corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

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

                # ---- ГОМОГРАФИЯ ----
                pt = np.array([[[center_x, center_y]]], dtype=np.float32)
                mapped_pt = cv2.perspectiveTransform(pt, homography_matrix)[0][0]

                mx, my = int(mapped_pt[0]), int(mapped_pt[1])

                ball_state.update(mx, my)
                event = detect_event(ball_state, mx, my)

                if event:
                    print("EVENT:", event)
                if ball_state.last_event:
                    cv2.putText(
                        top_view,
                        f"EVENT: {ball_state.last_event}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                # Фиксированный ID (один мяч)
                track_id = 0

                if track_id not in top_view_trajectories:
                    top_view_trajectories[track_id] = []

                top_view_trajectories[track_id].append((mx, my))
                if len(top_view_trajectories[track_id]) > MAX_TOP_VIEW_POINTS:
                    top_view_trajectories[track_id].pop(0)

                # Рисуем точку мяча
                cv2.circle(top_view, (mx, my), 10, (0, 255, 0), -1)

                # Рисуем траекторию
                pts_tv = np.array(top_view_trajectories[track_id], np.int32)
                if len(pts_tv) > 1:
                    cv2.polylines(
                        top_view,
                        [pts_tv],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=3
                    )
                zone = get_zone(mx, my)
                cv2.putText(
                    top_view,
                    f"Z{zone}",
                    (mx + 15, my - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )



            else:
                color = OTHER_COLOR
                label = f"Class {cls} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)
    # Уменьшаем для отображения
    frame_show = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    top_view_show = cv2.resize(top_view, (0, 0), fx=0.25, fy=0.25)

    cv2.imshow("Tracking (Camera View)", frame_show)
    cv2.imshow("Top View (Homography)", top_view_show)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Готово! Видео сохранено как output_tracking_only_ball.mp4")