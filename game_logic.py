# game_logic.py
import numpy as np
from enum import Enum
from collections import deque


class EventType(Enum):
    BOUNCE = "Bounce"
    HIT = "Hit"
    NET = "Net"
    OUT = "Out"
    SERVE = "Serve"
    POINT_WON = "Point"  # очко выиграно (мяч вышел за пределы после валидного отскока)


# Состояние мяча для отслеживания событий
class BallState:
    def __init__(self):
        # История позиций мяча в топ-вью (x, y)
        self.history = deque(maxlen=15)  # последние 15 кадров достаточно

        # Текущее состояние розыгрыща
        self.last_bounce_side = None  # 1 или 2 — сторона последнего отскока
        self.last_hit_side = None  # сторона последнего удара
        self.serve_detected = False
        self.last_event = None
        self.frames_since_bounce = 0
        self.frames_out = 0

        # Параметры стола (в пикселях после гомографии)
        self.TABLE_W = 2740
        self.TABLE_H = 1525
        self.MID_X = self.TABLE_W // 2
        self.MID_Y = self.TABLE_H // 2

        # Пороги
        self.BOUNCE_THRESHOLD = 30  # мин. изменение скорости по Y для отскока
        self.HIT_SPEED_THRESHOLD = 80  # резкое ускорение для удара
        self.NET_Y_TOLERANCE = 60  # сколько пикселей вокруг MID_Y считаем сеткой
        self.OUT_TOLERANCE = 100  # сколько кадров мяч может быть за столом до OUT

    def get_side(self, x):
        """Возвращает сторону стола: 1 (левая) или 2 (правая)"""
        return 1 if x < self.MID_X else 2

    def is_inside_table(self, x, y):
        """Проверяет, находится ли точка внутри стола"""
        return (0 <= x <= self.TABLE_W) and (0 <= y <= self.TABLE_H)

    def update(self, x, y):
        """
        Обновляет состояние мяча новой позицией (x, y) в топ-вью.
        Возвращает строку события, если оно произошло в этом кадре.
        """
        event = None

        if not self.is_inside_table(x, y):
            self.frames_out += 1
        else:
            self.frames_out = 0

        # Добавляем точку в историю
        self.history.append((x, y))

        if len(self.history) < 3:
            return None  # недостаточно данных

        # Вычисляем скорости между последними точками
        prev2, prev1, curr = self.history[-3], self.history[-2], self.history[-1]
        vx1 = prev1[0] - prev2[0]
        vy1 = prev1[1] - prev2[1]
        vx2 = curr[0] - prev1[0]
        vy2 = curr[1] - prev1[1]

        speed_prev = np.sqrt(vx1 ** 2 + vy1 ** 2)
        speed_curr = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # === Детекция отскока (Bounce) ===
        if abs(vy2 - vy1) > self.BOUNCE_THRESHOLD and speed_curr > 20:
            # Отскок обычно меняет направление по Y
            if vy1 * vy2 < 0:  # смена знака скорости по Y
                side = self.get_side(x)
                self.last_bounce_side = side
                self.frames_since_bounce = 0
                event = EventType.BOUNCE.value + f" (side {side})"
                self.last_event = event

        self.frames_since_bounce += 1

        # === Детекция удара ракеткой (Hit) ===
        acceleration = abs(speed_curr - speed_prev)
        if acceleration > self.HIT_SPEED_THRESHOLD and speed_curr > 50:
            side = self.get_side(x)
            self.last_hit_side = side
            event = EventType.HIT.value + f" (side {side})"
            self.last_event = event

        # === Детекция касания сетки (Net) ===
        if abs(y - self.MID_Y) < self.NET_Y_TOLERANCE:
            if speed_curr > 30 and abs(vx2) > 20:  # мяч пересекает среднюю линию
                event = EventType.NET.value
                self.last_event = event

        # === Детекция подачи (Serve) ===
        # Подача: мяч начинает движение из нижней половины одной стороны (y > MID_Y)
        if len(self.history) == 5 and not self.serve_detected:
            start_y = self.history[0][1]
            if start_y > self.MID_Y + 100:  # начиналась снизу
                start_side = self.get_side(self.history[0][0])
                if speed_curr > 30:
                    event = EventType.SERVE.value + f" (side {start_side})"
                    self.serve_detected = True
                    self.last_event = event

        # === Детекция выхода за стол (Out / Point) ===
        if self.frames_out > 8:  # мяч долго вне стола
            if self.last_bounce_side is not None:
                # Очко выигрывает игрок противоположной стороны
                winner_side = 1 if self.last_bounce_side == 2 else 2
                event = EventType.POINT_WON.value + f" (player {winner_side})"
                self.last_event = event
                # Сброс состояния для следующего розыгрыща
                self.reset_rally()

        if event:
            return event

        return None

    def reset_rally(self):
        """Сброс состояния после завершения розыгрыща"""
        self.history.clear()
        self.last_bounce_side = None
        self.last_hit_side = None
        self.serve_detected = False
        self.frames_since_bounce = 0
        self.frames_out = 0


# Функция-обёртка для удобного использования в основном скрипте
def detect_event(ball_state: BallState, x: int, y: int) -> str | None:
    """
    Вызывается каждый кадр с новой позицией мяча.
    Возвращает строку события или None.
    """
    return ball_state.update(x, y)