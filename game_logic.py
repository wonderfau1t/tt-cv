import numpy as np

TABLE_W = 2740
TABLE_H = 1525

MID_X = TABLE_W // 2
MID_Y = TABLE_H // 2

class BallState:
    def __init__(self, max_history=10, alpha=0.6):
        self.history = []
        self.filtered = None
        self.alpha = alpha
        self.max_history = max_history
        self.cooldown = 0
        self.last_event = None

    def update(self, x, y):
        if self.filtered is None:
            self.filtered = (x, y)
        else:
            fx, fy = self.filtered
            fx = self.alpha * x + (1 - self.alpha) * fx
            fy = self.alpha * y + (1 - self.alpha) * fy
            self.filtered = (fx, fy)

        self.history.append(self.filtered)
        if len(self.history) > self.max_history:
            self.history.pop(0)

def speed(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.hypot(dx, dy)

def velocity(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1]

def detect_bounce(history):
    if len(history) < 4:
        return False

    p1, p2, p3, p4 = history[-4:]

    vy1 = p2[1] - p1[1]
    vy2 = p3[1] - p2[1]

    s1 = speed(p1, p2)
    s2 = speed(p2, p3)

    return vy1 > 2 and vy2 < -2 and s2 < s1 * 0.8

def angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def detect_hit(history, angle_thresh=60, speed_gain=1.2):
    if len(history) < 4:
        return False

    p1, p2, p3, p4 = history[-4:]

    v1 = velocity(p1, p2)
    v2 = velocity(p3, p4)

    a = angle(v1, v2)
    s1 = speed(p1, p2)
    s2 = speed(p3, p4)

    return a > angle_thresh and s2 > s1 * speed_gain

def detect_net(history, mid_y,
               net_zone=30,
               min_speed_before=6,
               speed_drop_ratio=0.4,
               hang_frames=3):
    """
    Детекция касания сетки:
    мяч зависает и резко теряет скорость у линии сетки
    """

    if len(history) < hang_frames + 2:
        return False

    # Берём N последних точек
    pts = history[-(hang_frames + 2):]

    # Скорости
    speeds = [
        np.hypot(pts[i+1][0] - pts[i][0],
                 pts[i+1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ]

    # До сетки была нормальная скорость
    if speeds[0] < min_speed_before:
        return False

    # Почти остановка
    slow_frames = sum(s < speeds[0] * speed_drop_ratio for s in speeds[1:])

    # Точки рядом с сеткой
    near_net = all(abs(p[1] - mid_y) < net_zone for p in pts[1:-1])

    return slow_frames >= hang_frames and near_net


def inside_table(x, y):
    return 0 < x < TABLE_W and 0 < y < TABLE_H


def detect_out(x, y, w, h):
    return x < 0 or x > w or y < 0 or y > h

def detect_event(ball_state, x, y):
    if ball_state.cooldown > 0:
        ball_state.cooldown -= 1
        return None

    h = ball_state.history

    if detect_out(x, y, TABLE_W, TABLE_H):
        event = "OUT"
    elif detect_bounce(h) and inside_table(x, y):
        event = "BOUNCE"
    elif detect_net(ball_state.history, MID_Y):
        event = "NET"
    elif detect_hit(h):
        event = "HIT"
    else:
        event = None

    if event:
        ball_state.last_event = event
        ball_state.cooldown = 8
        return event

    return None
