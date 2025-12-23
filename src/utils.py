import json

import cv2
import numpy as np

TABLE_W = 2740
TABLE_H = 1525

MID_X = TABLE_W // 2
MID_Y = TABLE_H // 2


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
