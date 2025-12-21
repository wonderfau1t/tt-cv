import argparse
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


class ByteTracker:
    """
    Упрощенная реализация ByteTrack для трекинга мяча
    """

    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_id = 0

        # Треки
        self.tracks = []
        self.track_id = 0

        # Параметры для движения
        self.frame_rate = frame_rate
        self.max_time_lost = int(self.frame_rate / 30.0 * self.track_buffer)

    def update(self, dets):
        """
        Обновить трекер с новыми детекциями

        Args:
            dets: Детекции в формате [[x1, y1, x2, y2, conf], ...]

        Returns:
            Список треков в формате [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_id += 1

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(dets) > 0:
            # Фильтруем детекции порогу
            remain_inds = dets[:, 4] > self.track_thresh
            inds_low = dets[:, 4] > 0.1
            inds_high = dets[:, 4] < self.track_thresh

            remain_dets = dets[remain_inds]
            dets_low = dets[inds_low & inds_high]

            # Обновляем существующие треки
            if len(remain_dets) > 0:
                activated_starcks, refind_stracks, lost_stracks, removed_stracks = (
                    self._update_tracks(remain_dets)
                )
            else:
                # Если нет детекций выше порога, пытаемся найти с более низким порогом
                dets_low = dets[dets[:, 4] > 0.1]
                if len(dets_low) > 0:
                    activated_starcks, refind_stracks, lost_stracks, removed_stracks = (
                        self._update_tracks(dets_low, low_thresh=True)
                    )

        # Удаляем треки, которые были потеряны слишком долго
        for track in removed_stracks:
            self.tracks.remove(track)

        # Добавляем возвращенные треки обратно
        self.tracks.extend(activated_starcks)
        self.tracks.extend(refind_stracks)

        # Удаляем треки, которые были потеряны
        for track in lost_stracks:
            self.tracks.remove(track)

        # Возвращаем активные треки
        ret = []
        for track in self.tracks:
            if track.state == TrackState.Tracked:
                ret.append(np.concatenate([track.tlbr, [track.track_id]]).reshape(1, -1))

        if len(ret) > 0:
            return np.concatenate(ret)
        else:
            return np.empty((0, 5))

    def _update_tracks(self, detections, low_thresh=False):
        """
        Обновить треки на основе детекций

        Args:
            detections: Детекции в формате [[x1, y1, x2, y2, conf], ...]
            low_thresh: Использовать ли низкий порог для детекций

        Returns:
            activated_tracks, refind_tracks, lost_tracks, removed_tracks
        """
        activated_tracks = []
        refind_tracks = []
        lost_tracks = []
        removed_tracks = []

        if len(self.tracks) == 0:
            # Если нет существующих треков, создаем новые из детекций
            for det in detections:
                x1, y1, x2, y2, conf = det
                track = Track([x1, y1, x2, y2], conf, 0)
                track.activate(self.frame_id)
                activated_tracks.append(track)
            return activated_tracks, refind_tracks, lost_tracks, removed_tracks

        # Предсказываем позиции существующих треков
        for track in self.tracks:
            track.predict()

        # Вычисляем IoU между треками и детекциями
        track_bboxes = np.array([track.tlbr for track in self.tracks])
        det_bboxes = detections[:, :4]

        ious = self._compute_iou_matrix(track_bboxes, det_bboxes)

        # Сопоставляем треки с детекциями
        if ious.size > 0:
            # Используем простое сопоставление по максимальному IoU
            matched_indices = self._linear_assignment(ious)

            # Обновляем сопоставленные треки
            for track_idx, det_idx in matched_indices:
                track = self.tracks[track_idx]
                det = detections[det_idx]

                if ious[track_idx, det_idx] < self.match_thresh:
                    # Если IoU слишком мал, не обновляем трек
                    continue

                track.update(det[:4], det[4], self.frame_id)

            # Находим несопоставленные треки и детекции
            unmatched_tracks = [
                i for i in range(len(self.tracks)) if i not in matched_indices[:, 0]
            ]
            unmatched_dets = [i for i in range(len(detections)) if i not in matched_indices[:, 1]]
        else:
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = list(range(len(detections)))

        # Обрабатываем несопоставленные треки
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            # Если трек был активен слишком долго без обновления, помечаем как потерянный
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)
            else:
                track.mark_lost()
                lost_tracks.append(track)

        # Обрабатываем несопоставленные детекции
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            if det[4] > self.track_thresh or (low_thresh and det[4] > 0.1):
                # Создаем новый трек для детекции
                track = Track(det[:4], det[4], 0)
                track.activate(self.frame_id)
                activated_tracks.append(track)

        return activated_tracks, refind_tracks, lost_tracks, removed_tracks

    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """
        Вычислить матрицу IoU между двумя наборами bbox'ов
        """
        if bboxes1.size == 0 or bboxes2.size == 0:
            return np.empty((0, 0))

        # Вычисляем пересечения
        bboxes1 = np.expand_dims(bboxes1, 1)  # [N, 1, 4]
        bboxes2 = np.expand_dims(bboxes2, 0)  # [1, M, 4]

        xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h

        # Вычисляем площади
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

        union = area1 + area2 - inter
        iou = inter / (union + 1e-9)

        return iou

    def _linear_assignment(self, cost_matrix):
        """
        Простое сопоставление по минимальной стоимости (жадный алгоритм)
        """
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int)

        # Используем жадный алгоритм для сопоставления
        matched_indices = []
        cost_matrix_copy = cost_matrix.copy()

        # Найдем максимальное значение + 1 для "маскировки" уже сопоставленных
        max_val = cost_matrix_copy.max() + 1

        for _ in range(min(cost_matrix_copy.shape)):
            # Найдем максимальное значение IoU (минимальную стоимость в случае IoU)
            pos = np.unravel_index(np.argmax(cost_matrix_copy, axis=None), cost_matrix_copy.shape)
            matched_indices.append(pos)
            # "Маскируем" строку и столбец
            cost_matrix_copy[pos[0], :] = -1
            cost_matrix_copy[:, pos[1]] = -1

        return np.array(matched_indices, dtype=int)


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:
    def __init__(self, tlbr, score, track_id):
        self.tlbr = tlbr  # [x1, y1, x2, y2]
        self.score = score
        self.track_id = track_id
        self.state = TrackState.New
        self.is_activated = False
        self.frame_id = -1
        self.start_frame = -1
        self.mean = None
        self.covariance = None

    def activate(self, frame_id):
        self.track_id = self.next_id()
        self.is_activated = True
        self.state = TrackState.Tracked
        self.frame_id = frame_id
        self.start_frame = frame_id
        # Инициализируем простую модель движения
        self.mean = np.array([self.tlbr[0], self.tlbr[1], self.tlbr[2], self.tlbr[3], 0, 0])
        self.covariance = np.eye(6)

    def predict(self):
        # Простая модель движения (предполагаем, что объект движется с постоянной скоростью)
        if self.mean is not None:
            # Обновляем позицию на основе скорости
            dt = 1  # время между кадрами
            self.mean[0] += self.mean[4] * dt  # x1
            self.mean[1] += self.mean[5] * dt  # y1
            # x2, y2 не изменяются
            # обновляем скорость
            # (в реальной реализации здесь будет более сложная модель)

    def update(self, tlbr, score, frame_id):
        self.tlbr = tlbr
        self.score = score
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        # Обновляем модель движения
        if self.mean is not None:
            # Вычисляем разницу между предыдущей и текущей позицией
            dx = tlbr[0] - self.mean[0]
            dy = tlbr[1] - self.mean[1]
            # Обновляем скорость
            self.mean[4] = dx  # vx
            self.mean[5] = dy  # vy
            # Обновляем позицию
            self.mean[0] = tlbr[0]
            self.mean[1] = tlbr[1]
            self.mean[2] = tlbr[2]
            self.mean[3] = tlbr[3]

    def mark_lost(self):
        self.state = TrackState.Lost
        self.is_activated = False

    def mark_removed(self):
        self.state = TrackState.Removed
        self.is_activated = False

    @staticmethod
    def next_id():
        Track._count = getattr(Track, "_count", 0) + 1
        return Track._count


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

    # Инициализируем трекер мяча
    ball_tracker = ByteTracker(track_thresh=0.3, match_thresh=0.7)

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

        try:
            # Выполняем детекцию
            results = model(frame, verbose=False)
            detections = results[0].boxes
        except Exception as e:
            print(f"Error during detection: {e}")
            break

        # Обрабатываем детекции
        detections_list = []
        ball_positions = []  # Позиции мячей из детекции

        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh:
                # Сохраняем все детекции для трекинга
                detections_list.append([xmin, ymin, xmax, ymax, conf])

                # Сохраняем позиции мячей для отображения на гомографии
                if classname.lower() == "ball":
                    center_x = (xmin + xmax) // 2
                    center_y = (ymin + ymax) // 2
                    ball_positions.append((center_x, center_y))

        # Преобразуем в numpy массив
        if len(detections_list) > 0:
            detections_array = np.array(detections_list)
        else:
            detections_array = np.empty((0, 5))

        # Обновляем трекинг мяча только для мячей
        ball_detections = []
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()

            if conf > min_thresh and classname.lower() == "ball":
                ball_detections.append([xmin, ymin, xmax, ymax, conf])

        # Преобразуем в numpy массив
        if len(ball_detections) > 0:
            ball_dets = np.array(ball_detections)
            tracked_balls = ball_tracker.update(ball_dets)
        else:
            tracked_balls = np.empty((0, 5))

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

        # Рисуем все детекции
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

        # Рисуем треки мячей
        for track in tracked_balls:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)

            # Центр мяча
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Рисуем трек мяча (зеленый круг)
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Ball #{track_id}",
                (center_x + 15, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Создаем пустое изображение для вида сверху
        top_view = np.zeros((1525, 2740, 3), dtype=np.uint8)

        # Рисуем границы стола на виде сверху
        table_outline = np.array(dst_points, dtype=np.int32)
        cv2.polylines(top_view, [table_outline], True, (0, 25, 255), 2)

        # Отображаем отслеживаемые позиции мячей на виде сверху
        for track in tracked_balls:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Центр мяча
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Преобразуем координаты мяча в пространство гомографии
            ball_pos_homog = cv2.perspectiveTransform(
                np.array([[[center_x, center_y]]], dtype=np.float32), homography_matrix
            )[0][0]

            # Рисуем положение мяча на виде сверху
            cv2.circle(
                top_view,
                (int(ball_pos_homog[0]), int(ball_pos_homog[1])),
                10,
                (0, 255, 0),  # Зеленый для отслеживаемого мяча
                -1,
            )

            # Помечаем ID трека
            cv2.putText(
                top_view,
                f"#{int(track_id)}",
                (int(ball_pos_homog[0]) + 15, int(ball_pos_homog[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Изменяем размеры изображений для отображения
        display_frame = cv2.resize(frame, (960, 540))  # Уменьшаем размер основного окна
        display_top_view = cv2.resize(top_view, (960, 540))  # Уменьшаем размер вида сверху

        # Отображаем результаты
        cv2.imshow("YOLO Detection Results with Improved Tracking", display_frame)
        cv2.imshow("Top View with Improved Tracking", display_top_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Клавиша 'q' нажата, завершение программы...")
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
