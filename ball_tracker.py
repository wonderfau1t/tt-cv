from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics.models import YOLO


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

    def predict(self) -> Optional[Tuple[int, int]]:
        """
        Предсказать следующую позицию мяча

        Returns:
            Предсказанная позиция (x, y) или None если не инициализировано
        """
        if not self.initialized:
            return None

        prediction = self.kalman.predict()
        return int(prediction[0][0]), int(prediction[1][0])

    def update(self, measurement: Optional[Tuple[int, int]]) -> Tuple[int, int]:
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


class YOLOBallTracker:
    """
    Приложение для трекинга мяча с использованием YOLO и Kalman Filter
    """

    def __init__(self, model_path: str, video_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold

        # Загружаем модель YOLO
        self.model = YOLO(model_path, task="detect")

        # Инициализируем трекер мяча
        self.ball_tracker = BallTracker()

        # Открываем видео
        self.cap = cv2.VideoCapture(video_path)

        # Получаем параметры видео
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.dt = (
            1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        )  # стандартное значение если FPS не определен

        # Пересоздаем трекер с правильным значением dt
        self.ball_tracker = BallTracker(dt=self.dt)

        # Индекс класса мяча (из data.yaml видно, что ball - это индекс 0)
        self.ball_class_id = 0

    def detect_ball_with_yolo(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Детектировать мяч с помощью YOLO модели

        Args:
            frame: Входной кадр

        Returns:
            Позиция мяча (x, y) или None если мяч не найден
        """
        # Выполняем детекцию
        results = self.model(frame, verbose=False)

        # Извлекаем детекции
        detections = results[0].boxes

        # Находим все обнаруженные мячи
        ball_detections = []

        for i in range(len(detections)):
            # Получаем координаты ограничивающей рамки
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            # Получаем ID класса и уверенность
            class_idx = int(detections[i].cls.item())
            conf = detections[i].conf.item()

            # Проверяем, является ли объект мячом и уверенность выше порога
            if class_idx == self.ball_class_id and conf > self.confidence_threshold:
                # Центральная точка детекции
                center_x = int((xmin + xmax) / 2)
                center_y = int((ymin + ymax) / 2)

                ball_detections.append((center_x, center_y, conf))

        # Если найдено несколько мячей, выбираем наиболее уверенную детекцию
        if ball_detections:
            # Сортируем по уверенности (в порядке убывания)
            ball_detections.sort(key=lambda x: x[2], reverse=True)
            # Возвращаем центральную точку самого уверенного обнаружения
            return ball_detections[0][:2]  # возвращаем только x, y

        return None

    def run(self):
        """
        Запустить трекинг мяча
        """
        frame_count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Детектируем мяч с помощью YOLO
            ball_pos = self.detect_ball_with_yolo(frame)

            # Обновляем трекер с новым измерением
            tracked_pos = self.ball_tracker.update(ball_pos)

            # Рисуем результаты
            if ball_pos is not None:
                # Отрисовываем измеренное положение (синий)
                cv2.circle(frame, ball_pos, 10, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    "Detected",
                    (ball_pos[0] + 15, ball_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            # Отрисовываем отслеживаемое положение (зеленый)
            cv2.circle(frame, tracked_pos, 10, (0, 25, 0), 2)
            cv2.putText(
                frame,
                "Tracked",
                (tracked_pos[0] + 15, tracked_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Рисуем траекторию
            for i in range(1, len(self.ball_tracker.trajectory)):
                cv2.line(
                    frame,
                    self.ball_tracker.trajectory[i - 1],
                    self.ball_tracker.trajectory[i],
                    (0, 255, 255),
                    2,
                )

            # Отображаем номер кадра
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Отображаем кадр
            cv2.imshow("Ball Tracking with YOLO", frame)

            frame_count += 1

            # Нажмите 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Освобождаем ресурсы
        self.cap.release()
        cv2.destroyAllWindows()

    def run_with_output(self, output_path: Optional[str] = None):
        """
        Запустить трекинг мяча с возможностью сохранения видео

        Args:
            output_path: Путь для сохранения результата (опционально)
        """
        frame_count = 0

        # Подготовка записи видео
        out = None
        if output_path:
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(
                output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Детектируем мяч с помощью YOLO
            ball_pos = self.detect_ball_with_yolo(frame)

            # Обновляем трекер с новым измерением
            tracked_pos = self.ball_tracker.update(ball_pos)

            # Рисуем результаты
            if ball_pos is not None:
                # Отрисовываем измеренное положение (синий)
                cv2.circle(frame, ball_pos, 10, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    "Detected",
                    (ball_pos[0] + 15, ball_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

            # Отрисовываем отслеживаемое положение (зеленый)
            cv2.circle(frame, tracked_pos, 10, (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Tracked",
                (tracked_pos[0] + 15, tracked_pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Рисуем траекторию
            for i in range(1, len(self.ball_tracker.trajectory)):
                cv2.line(
                    frame,
                    self.ball_tracker.trajectory[i - 1],
                    self.ball_tracker.trajectory[i],
                    (0, 255, 255),
                    2,
                )

            # Отображаем номер кадра и статус
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if ball_pos is not None:
                cv2.putText(
                    frame,
                    "Status: DETECTED",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    "Status: PREDICTED",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )

            # Сохраняем кадр если указан путь вывода
            if out:
                out.write(frame)

            # Отображаем кадр
            cv2.imshow("Ball Tracking with YOLO", frame)

            frame_count += 1

            # Нажмите 'q' для выхода
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Освобождаем ресурсы
        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def main():
    """
    Основная функция для запуска трекинга мяча
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Track ball in video using YOLO and Kalman Filter")
    parser.add_argument(
        "--model", help="Path to YOLO model file", default="model/ppv_yolo11s_based.pt"
    )
    parser.add_argument("--video", help="Path to input video file", required=True)
    parser.add_argument("--output", help="Path to output video file (optional)", default=None)
    parser.add_argument(
        "--confidence", help="Confidence threshold for ball detection", type=float, default=0.5
    )

    args = parser.parse_args()

    # Проверяем существование файлов
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return

    # Создаем трекер
    tracker_app = YOLOBallTracker(
        model_path=args.model, video_path=args.video, confidence_threshold=args.confidence
    )

    # Запускаем трекинг
    tracker_app.run_with_output(output_path=args.output)


if __name__ == "__main__":
    main()
