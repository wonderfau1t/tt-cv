import numpy as np

TABLE_W = 2740
TABLE_H = 1525

MID_X = TABLE_W // 2
MID_Y = TABLE_H // 2

LEFT = 1
RIGHT = 2

POINTS_TO_WIN_GAME = 11
MIN_DIFF = 2


def side_of_table(x):
    return LEFT if x < MID_X else RIGHT

from enum import Enum

class RallyState(Enum):
    IDLE = 0          # нет розыгрыша
    SERVE = 1         # подача
    FLYING = 2        # мяч в игре
    BOUNCED = 3       # был отскок
    FINISHED = 4      # розыгрыш завершён

class RallyFSM:
    def __init__(self):
        self.state = RallyState.IDLE
        self.server = None
        self.last_hitter = None
        self.last_bounce_side = None

    def reset(self):
        self.state = RallyState.IDLE
        self.server = None
        self.last_hitter = None
        self.last_bounce_side = None

    def step(self, event, side):
        if event is None:
            return None

        # ---------- IDLE (ожидание подачи) ----------
        if self.state == RallyState.IDLE:
            if event == "HIT":
                self.server = side
                self.last_hitter = side
                self.state = RallyState.FLYING
            return None

        # ---------- FLYING (мяч в воздухе) ----------
        if self.state == RallyState.FLYING:

            if event == "HIT":
                # двойной удар
                if side == self.last_hitter:
                    self.state = RallyState.FINISHED
                    return side
                self.last_hitter = side
                return None

            if event == "BOUNCE":
                self.last_bounce_side = side
                self.state = RallyState.BOUNCED
                return None

            if event in ("NET", "OUT"):
                self.state = RallyState.FINISHED
                return self.last_hitter

        # ---------- BOUNCED ----------
        if self.state == RallyState.BOUNCED:

            if event == "BOUNCE":
                # двойной отскок
                self.state = RallyState.FINISHED
                return self.last_hitter

            if event == "HIT":
                self.last_hitter = side
                self.state = RallyState.FLYING
                return None

            if event in ("NET", "OUT"):
                self.state = RallyState.FINISHED
                return self.last_hitter

        return None


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

def detect_bounce(history,
                  min_vy=2,
                  speed_drop=0.75,
                  max_x_change_ratio=0.3):
    if len(history) < 4:
        return False

    p1, p2, p3, p4 = history[-4:]

    v1 = velocity(p1, p2)
    v2 = velocity(p2, p3)

    # смена направления по Y
    y_flip = v1[1] * v2[1] < -min_vy

    # X почти не меняется
    x_stable = abs(v2[0] - v1[0]) < abs(v1[0]) * max_x_change_ratio

    s1 = speed(p1, p2)
    s2 = speed(p2, p3)

    speed_drop_ok = s2 < s1 * speed_drop

    return y_flip and x_stable and speed_drop_ok


def angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def detect_hit(history,
               min_vx=3,
               speed_gain=1.2):
    if len(history) < 4:
        return False

    p1, p2, p3, p4 = history[-4:]

    v1 = velocity(p1, p2)
    v2 = velocity(p3, p4)

    # смена направления по X
    x_flip = v1[0] * v2[0] < -min_vx

    # Y не обязан меняться
    s1 = speed(p1, p2)
    s2 = speed(p3, p4)

    speed_gain_ok = s2 > s1 * speed_gain

    return x_flip and speed_gain_ok


def detect_net(history,
               mid_x,
               net_zone=25,
               min_speed_before=6,
               speed_drop_ratio=0.4,
               hang_frames=3):

    if len(history) < hang_frames + 2:
        return False

    pts = history[-(hang_frames + 2):]

    speeds = [
        np.hypot(pts[i+1][0] - pts[i][0],
                 pts[i+1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    ]

    if speeds[0] < min_speed_before:
        return False

    slow_frames = sum(s < speeds[0] * speed_drop_ratio for s in speeds[1:])

    near_net = all(abs(p[0] - mid_x) < net_zone for p in pts[1:-1])

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
    elif detect_net(ball_state.history, MID_X):
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

class GameState:
    def __init__(self):
        self.score = {LEFT: 0, RIGHT: 0}
        self.fsm = RallyFSM()

    def on_event(self, event, x):
        side = side_of_table(x)
        loser = self.fsm.step(event, side)

        if loser:
            winner = RIGHT if loser == LEFT else LEFT
            self.score[winner] += 1
            self.fsm.reset()
            return winner

        return None

class Game:
    def __init__(self):
        self.score = {LEFT: 0, RIGHT: 0}
        self.finished = False
        self.winner = None

    def add_point(self, side):
        if self.finished:
            return None

        self.score[side] += 1

        l, r = self.score[LEFT], self.score[RIGHT]

        if (l >= POINTS_TO_WIN_GAME or r >= POINTS_TO_WIN_GAME) and abs(l - r) >= MIN_DIFF:
            self.finished = True
            self.winner = LEFT if l > r else RIGHT
            return self.winner

        return None

class Match:
    def __init__(self, best_of=5):
        """
        best_of = 5 → до 3 побед
        best_of = 7 → до 4 побед
        """
        self.games_to_win = best_of // 2 + 1
        self.games_won = {LEFT: 0, RIGHT: 0}
        self.current_game = Game()
        self.finished = False
        self.winner = None

    def on_rally_end(self, winner):
        if self.finished:
            return None

        game_winner = self.current_game.add_point(winner)

        if game_winner:
            self.games_won[game_winner] += 1

            if self.games_won[game_winner] >= self.games_to_win:
                self.finished = True
                self.winner = game_winner
                return self.winner

            # новый сет
            self.current_game = Game()

        return None
