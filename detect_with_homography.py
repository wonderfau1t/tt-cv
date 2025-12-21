import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics.models import YOLO


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", help="Path to YOLO model file", default="model/ppv_yolo11s_based.pt"
    )
    parser.add_argument("--source", help="Video source", default="videos/test_1.mp4")
    parser.add_argument(
        "--corners", help="JSON file with table corners", default="table_corners.json"
    )
    parser.add_argument("--thresh", help="Minimum confidence threshold", default=0.5)

    args = parser.parse_args()

    model_path = args.model
    video_source = args.source
    corners_path = args.corners
    min_thresh = float(args.thresh)

    # Проверяем существование модели
    if not os.path.exists(model_path):
        print(f"ERROR: Model path is invalid or model was not found: {model_path}")
        sys.exit(0)

    # Загружаем модель
    model = YOLO(model_path, task="detect")
    labels = model.names

    # Загружаем координаты углов стола
    src_corners = load_table_corners(corners_path)
    print(f"Загружены углы стола: {src_corners}")

    # Вычисляем матрицу гомографии
    homography_matrix, dst_points = compute_homography_matrix(src_corners)
    print("Матрица гомографии вычислена")

    # Открываем видео
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {video_source}")
        sys.exit(0)

    # Проверяем, что матрица гомографии корректно вычислена
    if homography_matrix is None:
        print("ERROR: Could not compute homography matrix")
        sys.exit(0)

    # Устанавливаем цвета для bounding boxes
    bbox_colors = [
        (164, 120, 87),
        (68, 148, 228),
        (93, 97, 209),
        (178, 182, 133),
        (88, 159, 106),
        (96, 202, 231),
        (159, 124, 168),
        (169, 162, 241),
        (98, 118, 150),
        (172, 176, 184),
    ]

    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of the video file. Exiting program.")
            break

        try:
            # Выполняем детекцию
            results = model(frame, verbose=False)
            detections = results[0].boxes
        except Exception as e:
            print(f"Error during detection: {e}")
            break

        # Выполняем детекцию
        results = model(frame, verbose=False)
        detections = results[0].boxes

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

        # Обрабатываем детекции
        ball_positions = []
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                label = f"{classname}: {int(conf * 100)}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                # Сохраняем позиции мячей для отображения на гомографии
                if classname.lower() == "ball":
                    ball_positions.append(((xmin + xmax) // 2, (ymin + ymax) // 2))

        # Создаем пустое изображение для вида сверху
        top_view = np.zeros((1525, 2740, 3), dtype=np.uint8)

        # Рисуем границы стола на виде сверху
        table_outline = np.array(dst_points, dtype=np.int32)
        cv2.polylines(top_view, [table_outline], True, (0, 255, 255), 2)

        # Отображаем позиции мячей на виде сверху
        for ball_pos in ball_positions:
            # Преобразуем координаты мяча в пространство гомографии
            ball_pos_homog = cv2.perspectiveTransform(
                np.array([[[ball_pos[0], ball_pos[1]]]], dtype=np.float32), homography_matrix
            )[0][0]

            # Рисуем положение мяча и его bounding box на виде сверху
            cv2.circle(
                top_view,
                (int(ball_pos_homog[0]), int(ball_pos_homog[1])),
                10,
                (0, 255, 0),
                -1,
            )

            # Рисуем bounding box для мяча
            cv2.rectangle(
                top_view,
                (int(ball_pos_homog[0]) - 15, int(ball_pos_homog[1]) - 15),
                (int(ball_pos_homog[0]) + 15, int(ball_pos_homog[1]) + 15),
                (0, 255, 0),
                2,
            )

        # Изменяем размеры изображений для отображения
        display_frame = cv2.resize(frame, (960, 540))  # Уменьшаем размер основного окна
        display_top_view = cv2.resize(top_view, (960, 540))  # Уменьшаем размер вида сверху

        # Отображаем результаты
        cv2.imshow("YOLO Detection Results", display_frame)
        cv2.imshow("Top View", display_top_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Клавиша 'q' нажата, завершение программы...")
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
