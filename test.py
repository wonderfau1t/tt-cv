import json
from pathlib import Path
 
import cv2
 
 
class TableAnnotator:
    """
    Класс для ручной разметки стола на видео
    """
 
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
 
        # Переменные для ручной разметки
        self.points = []  # Массив для хранения 4 точек
        self.current_point = None  # Текущая точка
 
        # Имя окна
        self.window_name = "Ручная разметка стола - Нажмите 4 точки угла стола"
 
    def mouse_callback(self, event, x, y, flags, param):
        """
        Обработчик событий мыши для ручной разметки
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Если у нас еще не 4 точки, добавляем новую точку
            if len(self.points) < 4:
                self.points.append((x, y))
 
                print(f"Добавлена точка {len(self.points)}: ({x}, {y})")
 
                # Если набрали 4 точки, показываем инструкции для завершения
                if len(self.points) == 4:
                    print("Добавлены 4 точки стола:")
                    for i, point in enumerate(self.points):
                        print(f" Точка {i + 1}: {point}")
                    print("Нажмите 's' для сохранения разметки")
                    print("Нажмите 'r' для сброса точек")
                    print("Нажмите 'u' для отмены последней точки")
 
    def annotate_first_frame(self):
        """
        Ручная разметка стола на первом кадре видео
        """
        # Устанавливаем на первый кадр
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
 
        if not ret:
            print("Ошибка: Не удалось прочитать первый кадр видео")
            return None
 
        # Создаем окно и регистрируем обработчик мыши
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
 
        print("Инструкции:")
        print("- Нажмите 4 раза в углах стола (по часовой стрелке или против)")
        print("- Порядок точек: левый-верхний, правый-верхний, правый-нижний, левый-нижний")
        print("- Нажмите 's' для сохранения разметки")
        print("- Нажмите 'r' для сброса всех точек")
        print("- Нажмите 'u' для отмены последней точки")
        print("- Нажмите 'q' для выхода без сохранения")
        print()
 
        while True:
            display_frame = frame.copy()
 
            # Рисуем точки и соединяем их линиями
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 8, (0, 25, 0), -1)
                cv2.putText(
                    display_frame,
                    f"{i + 1}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
 
            # Рисуем линии между точками, если есть хотя бы 2 точки
            if len(self.points) >= 2:
                for i in range(len(self.points) - 1):
                    cv2.line(display_frame, self.points[i], self.points[i + 1], (0, 255, 0), 2)
 
                # Если есть 4 точки, соединяем последнюю с первой
                if len(self.points) == 4:
                    cv2.line(display_frame, self.points[3], self.points[0], (0, 255, 0), 2)
 
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("s"):  # Сохранить разметку
                if len(self.points) == 4:
                    # Сохраняем 4 точки как полигональную разметку
                    table_polygon = self.points
 
                    # Сохраняем разметку в JSON файл
                    annotation_data = {
                        "video_path": str(self.video_path),
                        "table_polygon": table_polygon,  # 4 точки полигона
                        "frame_number": 0,  # Первый кадр
                    }
 
                    # Сохраняем в файл рядом с видео
                    video_path_obj = Path(self.video_path)
                    annotation_file = video_path_obj.with_suffix(".table_annotation.json")
 
                    with open(annotation_file, "w", encoding="utf-8") as f:
                        json.dump(annotation_data, f, indent=2, ensure_ascii=False)
 
                    print(f"Разметка сохранена в {annotation_file}")
                    print(f"Координаты стола (4 точки): {table_polygon}")
                    break
                else:
                    print(f"Нужно отметить 4 точки, отмечено: {len(self.points)}")
 
            elif key == ord("r"):  # Сбросить все точки
                self.points = []
                print("Все точки сброшены")
 
            elif key == ord("u"):  # Отменить последнюю точку
                if self.points:
                    removed_point = self.points.pop()
                    print(f"Последняя точка {removed_point} удалена")
 
            elif key == ord("q"):  # Выйти без сохранения
                print("Выход без сохранения разметки")
                break
 
        cv2.destroyAllWindows()
        self.cap.release()
        return len(self.points) == 4
 
    def annotate_specific_frame(self, frame_number):
        """
        Ручная разметка стола на определенном кадре
        """
        # Устанавливаем на указанный кадр
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
 
        if not ret:
            print(f"Ошибка: Не удалось прочитать кадр #{frame_number}")
            return None
 
        # Создаем окно и регистрируем обработчик мыши
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
 
        print(f"Инструкции для кадра #{frame_number}:")
        print("- Нажмите 4 раза в углах стола (по часовой стрелке или против)")
        print("- Порядок точек: левый-верхний, правый-верхний, правый-нижний, левый-нижний")
        print("- Нажмите 's' для сохранения разметки")
        print("- Нажмите 'r' для сброса всех точек")
        print("- Нажмите 'u' для отмены последней точки")
        print("- Нажмите 'q' для выхода без сохранения")
        print()
 
        while True:
            display_frame = frame.copy()
 
            # Рисуем точки и соединяем их линиями
            for i, point in enumerate(self.points):
                cv2.circle(display_frame, point, 8, (0, 255, 0), -1)
                cv2.putText(
                    display_frame,
                    f"{i + 1}",
                    (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
 
            # Рисуем линии между точками, если есть хотя бы 2 точки
            if len(self.points) >= 2:
                for i in range(len(self.points) - 1):
                    cv2.line(display_frame, self.points[i], self.points[i + 1], (0, 255, 0), 2)
 
                # Если есть 4 точки, соединяем последнюю с первой
                if len(self.points) == 4:
                    cv2.line(display_frame, self.points[3], self.points[0], (0, 255, 0), 2)
 
            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("s"):  # Сохранить разметку
                if len(self.points) == 4:
                    # Сохраняем 4 точки как полигональную разметку
                    table_polygon = self.points
 
                    # Сохраняем разметку в JSON файл
                    annotation_data = {
                        "video_path": str(self.video_path),
                        "table_polygon": table_polygon,  # 4 точки полигона
                        "frame_number": frame_number,
                    }
 
                    # Сохраняем в файл рядом с видео
                    video_path_obj = Path(self.video_path)
                    annotation_file = video_path_obj.with_suffix(".table_annotation.json")
 
                    with open(annotation_file, "w", encoding="utf-8") as f:
                        json.dump(annotation_data, f, indent=2, ensure_ascii=False)
 
                    print(f"Разметка сохранена в {annotation_file}")
                    print(f"Координаты стола (4 точки): {table_polygon}")
                    break
                else:
                    print(f"Нужно отметить 4 точки, отмечено: {len(self.points)}")
 
            elif key == ord("r"):  # Сбросить все точки
                self.points = []
                print("Все точки сброшены")
 
            elif key == ord("u"):  # Отменить последнюю точку
                if self.points:
                    removed_point = self.points.pop()
                    print(f"Последняя точка {removed_point} удалена")
 
            elif key == ord("q"):  # Выйти без сохранения
                print("Выход без сохранения разметки")
                break
 
        cv2.destroyAllWindows()
        self.cap.release()
        return len(self.points) == 4
 
 
def main():
    import argparse
 
    parser = argparse.ArgumentParser(description="Ручная разметка стола на видео")
    parser.add_argument("--video", type=str, default="videos/test_1.mp4", help="Путь к видео файлу")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Номер кадра для разметки (по умолчанию 0 - первый кадр)",
    )
 
    args = parser.parse_args()
 
    annotator = TableAnnotator(args.video)
 
    if args.frame == 0:
        annotator.annotate_first_frame()
    else:
        annotator.annotate_specific_frame(args.frame)
 
 
if __name__ == "__main__":
    main()
 