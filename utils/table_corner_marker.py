import argparse
import json
from pathlib import Path

import cv2
import numpy as np


class TableCornerMarker:
    """
    Класс для отображения и маркировки углов стола для настольного тенниса
    """

    def __init__(self, video_path, output_json_path):
        self.video_path = video_path
        self.output_json_path = output_json_path
        self.corners = []  # Список для хранения координат углов
        self.max_corners = 4  # Максимальное количество углов (4 угла стола)
        self.window_name = "Отметьте углы стола для настольного тенниса"

    def mouse_callback(self, event, x, y, flags, param):
        """
        Обработчик событий мыши для отметки углов
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < self.max_corners:
                self.corners.append((x, y))
                print(f"Добавлена точка {len(self.corners)}: ({x}, {y})")

                # Отобразим точку на кадре
                frame_copy = self.current_frame.copy()
                for i, corner in enumerate(self.corners):
                    cv2.circle(frame_copy, corner, 10, (0, 255, 0), -1)
                    cv2.putText(
                        frame_copy,
                        str(i + 1),
                        (corner[0] + 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                cv2.imshow(self.window_name, frame_copy)

                # Если все 4 угла отмечены, завершаем процесс
                if len(self.corners) == self.max_corners:
                    print("Все 4 угла отмечены!")
                    self.save_corners()
                    cv2.destroyWindow(self.window_name)

    def save_corners(self):
        """
        Сохранение координат углов в JSON файл
        """
        corners_data = {
            "video_path": str(self.video_path),
            "corners": [{"x": corner[0], "y": corner[1]} for corner in self.corners],
            "description": "Координаты углов стола для настольного тенниса (верхний левый, верхний правый, нижний правый, нижний левый)",
        }

        # Определение порядка углов (предполагаем стандартный порядок)
        # Для настольного тенниса: [верх-лево, верх-право, низ-право, низ-лево]
        if len(self.corners) == 4:
            sorted_corners = self.sort_corners(self.corners)
            corners_data["corners_sorted"] = [
                {"x": corner[0], "y": corner[1]} for corner in sorted_corners
            ]

        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(corners_data, f, indent=2, ensure_ascii=False)

        print(f"Координаты углов сохранены в {self.output_json_path}")

    def sort_corners(self, corners):
        """
        Сортировка углов в порядке: верх-лево, верх-право, низ-право, низ-лево
        """
        # Преобразуем в массив numpy для удобной сортировки
        pts = np.array(corners, dtype="float32")

        # Сортируем точки по сумме координат (верхние левые будут меньше)
        # и разности координат (для различения диагоналей)
        rect = np.zeros((4, 2), dtype="float32")

        # Сумма координат: [x+y] - верхние левые будут меньше, нижние правые больше
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # верх-лево (минимальная сумма)
        rect[2] = pts[np.argmax(s)]  # низ-право (максимальная сумма)

        # Разность координат: [x-y] - верх-право будет больше, низ-лево меньше
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmax(diff)]  # верх-право (максимальная разность)
        rect[3] = pts[np.argmin(diff)]  # низ-лево (минимальная разность)

        return rect.astype(int).tolist()

    def run(self):
        """
        Запуск процесса маркировки углов
        """
        # Открываем видеофайл
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видеофайл: {self.video_path}")

        # Получаем первый кадр
        ret, self.current_frame = cap.read()
        if not ret:
            raise ValueError("Не удалось прочитать кадр из видеофайла")

        # Создаем окно и устанавливаем обработчик событий мыши
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("Инструкция:")
        print("- Нажмите левой кнопкой мыши на 4 угла стола для настольного тенниса")
        print("- Порядок: верх-лево, верх-право, низ-право, низ-лево (по вашему усмотрению)")
        print("- После отметки 4 углов окно закроется автоматически")
        print("- Нажмите 'q' для выхода без сохранения")
        print("- Нажмите 'r' для сброса и повторной маркировки")

        while True:
            # Отображаем текущий кадр с уже отмеченными точками
            display_frame = self.current_frame.copy()

            # Рисуем уже отмеченные точки
            for i, corner in enumerate(self.corners):
                cv2.circle(display_frame, corner, 10, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    str(i + 1),
                    (corner[0] + 10, corner[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Если отмечено менее 4 углов, показываем инструкцию
            if len(self.corners) < self.max_corners:
                remaining = self.max_corners - len(self.corners)
                cv2.putText(
                    display_frame,
                    f"Осталось отметить: {remaining} углов",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Если все углы отмечены, выходим
            if len(self.corners) == self.max_corners:
                break

            # Клавиша 'q' для выхода без сохранения
            if key == ord("q"):
                print("Выход без сохранения...")
                break

            # Клавиша 'r' для сброса точек
            if key == ord("r"):
                self.corners = []
                print("Точки сброшены, можно начать заново")

        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Отметка углов стола для настольного тенниса на видео"
    )
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    parser.add_argument(
        "--output", required=True, help="Путь к выходному JSON файлу с координатами"
    )

    args = parser.parse_args()

    # Проверяем существование видеофайла
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Ошибка: видеофайл не найден: {args.video}")
        return

    # Создаем директорию для выходного файла если её нет
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Запускаем маркер углов
    marker = TableCornerMarker(video_path, output_path)
    marker.run()


if __name__ == "__main__":
    main()
