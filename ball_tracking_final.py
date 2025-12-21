import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


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


class BallTracker:
    """
    Класс для трекинга мяча с использованием OpenCV и Kalman Filter
    """

    def __init__(
        self, dt: float = 1.0, process_noise: float = 1e-2, measurement_noise: float = 1e-1
    ):
        """
        Инициализация трекера мяча

        Args:
            dt: Временной шаг между кадрами
            process_noise: Уровень шума процесса
            measurement_noise: Уровень шума измерений
        """
        # Размерности векторов состояния
        # x, y, vx, vy (положение и скорость)
        self.kalman = cv2.KalmanFilter(4, 2)

        # Матрица перехода состояния (для модели постоянной скорости)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )

        # Матрица измерения (наблюдаем только положение)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)

        # Матрица процессного шума
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Матрица шума измерений
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Матрица ошибки априори
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Предыдущее состояние
        self.prev_measurement = None
        self.initialized = False

        # Траектория мяча
        self.trajectory = []
        self.max_trajectory_length = 100

    def predict(self):
        """
        Предсказать следующую позицию мяча

        Returns:
            Предсказанная позиция (x, y) или None если не инициализировано
        """
        if not self.initialized:
            return None

        prediction = self.kalman.predict()
        return int(prediction[0][0]), int(prediction[1][0])

    def update(self, measurement):
        """
        Обновить трекер с новым измерением

        Args:
            measurement: Позиция мяча (x, y) или None если мяч не обнаружен

        Returns:
            Обновленная позиция (x, y)
        """
        if measurement is not None:
            # Если есть измерение
            x, y = measurement
            measured_pos = np.array([[np.float32(x)], [np.float32(y)]])

            if not self.initialized:
                # Инициализация при первом измерении
                self.kalman.statePre = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.kalman.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.initialized = True
            else:
                # Коррекция с помощью измерения
                self.kalman.correct(measured_pos)

            self.prev_measurement = measurement
            # Добавляем текущую позицию в траекторию
            self._add_to_trajectory(x, y)
            return x, y
        else:
            # Если нет измерения, используем предсказание
            if self.initialized:
                prediction = self.kalman.predict()
                pred_x, pred_y = int(prediction[0][0]), int(prediction[1][0])
                # Добавляем предсказанную позицию в траекторию
                self._add_to_trajectory(pred_x, pred_y)
                return pred_x, pred_y
            else:
                return 0, 0

    def _add_to_trajectory(self, x: int, y: int):
        """
        Добавить точку в траекторию
        """
        self.trajectory.append((x, y))
        if len(self.trajectory) > self.max_trajectory_length:
            self.trajectory.pop(0)

    def reset(self):
        """
        Сбросить состояние трекера
        """
        self.prev_measurement = None
        self.initialized = False
        self.trajectory = []


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

    # Инициализируем трекер мяча
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    ball_tracker = BallTracker(dt=dt)

    # Устанавливаем цвета для bounding boxes
    bbox_colors = [
        (164, 120, 87),
        (68, 148, 28),
        (93, 97, 209),
        (178, 182, 133),
        (8, 159, 106),
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

        # Выполняем детекцию с трекингом
        results = model.track(frame, persist=True, verbose=False)
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
        cv2.polylines(frame, [pts], True, (0, 25, 255), 2)

        # Обрабатываем детекции
        ball_positions = []  # Позиции мячей из детекции
        detected_ball_pos = None  # Текущая позиция мяча для трекинга

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                color = bbox_colors[classidx % 10]

                # Рисуем bounding box
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

                # Проверяем, является ли объект мячом
                if classname.lower() == "ball":
                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2
                    ball_positions.append((center_x, center_y))

                    # Используем только одну детекцию мяча для трекинга (берем первую)
                    if detected_ball_pos is None:
                        detected_ball_pos = (center_x, center_y)

        # Обновляем трекинг мяча
        tracked_ball_pos = ball_tracker.update(detected_ball_pos)

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
                (25, 0, 0),  # Синий для детектированного мяча
                -1,
            )

            # Рисуем bounding box для мяча
            cv2.rectangle(
                top_view,
                (int(ball_pos_homog[0]) - 15, int(ball_pos_homog[1]) - 15),
                (int(ball_pos_homog[0]) + 15, int(ball_pos_homog[1]) + 15),
                (25, 0, 0),  # Синий для детектированного мяча
                2,
            )

        # Отображаем отслеживаемое положение мяча на виде сверху
        tracked_pos_homog = cv2.perspectiveTransform(
            np.array([[[tracked_ball_pos[0], tracked_ball_pos[1]]]], dtype=np.float32),
            homography_matrix,
        )[0][0]

        # Рисуем отслеживаемое положение мяча на виде сверху
        cv2.circle(
            top_view,
            (int(tracked_pos_homog[0]), int(tracked_pos_homog[1])),
            10,
            (0, 255, 0),  # Зеленый для отслеживаемого мяча
            2,
        )

        # Рисуем траекторию отслеживания на виде сверху
        if len(ball_tracker.trajectory) > 1:
            for i in range(1, len(ball_tracker.trajectory)):
                prev_pos = ball_tracker.trajectory[i - 1]
                curr_pos = ball_tracker.trajectory[i]

                # Преобразуем точки траектории в пространство гомографии
                prev_pos_homog = cv2.perspectiveTransform(
                    np.array([[[prev_pos[0], prev_pos[1]]]], dtype=np.float32), homography_matrix
                )[0][0]

                curr_pos_homog = cv2.perspectiveTransform(
                    np.array([[[curr_pos[0], curr_pos[1]]]], dtype=np.float32), homography_matrix
                )[0][0]

                # Рисуем линию траектории
                cv2.line(
                    top_view,
                    (int(prev_pos_homog[0]), int(prev_pos_homog[1])),
                    (int(curr_pos_homog[0]), int(curr_pos_homog[1])),
                    (0, 255, 255),  # Желтый для траектории
                    2,
                )

        # Также рисуем трекинг на основном кадре
        if detected_ball_pos is not None:
            # Рисуем детектированное положение мяча (синий)
            cv2.circle(frame, detected_ball_pos, 10, (255, 0, 0), -1)
            cv2.putText(
                frame,
                "Detected",
                (detected_ball_pos[0] + 15, detected_ball_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        # Рисуем отслеживаемое положение мяча (зеленый)
        cv2.circle(frame, tracked_ball_pos, 10, (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Tracked",
            (tracked_ball_pos[0] + 15, tracked_ball_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Рисуем траекторию на основном кадре
        for i in range(1, len(ball_tracker.trajectory)):
            cv2.line(
                frame, ball_tracker.trajectory[i - 1], ball_tracker.trajectory[i], (0, 255, 255), 2
            )

        # Изменяем размеры изображений для отображения
        display_frame = cv2.resize(frame, (960, 540))  # Уменьшаем размер основного окна
        display_top_view = cv2.resize(top_view, (960, 540))  # Уменьшаем размер вида сверху

        # Отображаем результаты
        cv2.imshow("YOLO Detection Results with Tracking", display_frame)
        cv2.imshow("Top View with Tracking", display_top_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Клавиша 'q' нажата, завершение программы...")
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
