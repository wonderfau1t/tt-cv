# tt_processor.py

import json
import cv2
import numpy as np
from ultralytics.models import YOLO
from game_logic import *

TABLE_W = 2740
TABLE_H = 1525

MID_X = TABLE_W // 2
MID_Y = TABLE_H // 2

BALL_COLOR = (0, 255, 0)
OTHER_COLOR = (255, 0, 0)
TRAJECTORY_COLOR = (0, 255, 255)

TRAJECTORY_THICKNESS = 3
MAX_TRAJECTORY_POINTS = 20
MAX_TOP_VIEW_POINTS = 50


def load_table_corners(json_path: str) -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array([[c["x"], c["y"]] for c in data["corners"]], dtype=np.float32)


def compute_homography_matrix(src_points: np.ndarray):
    dst_points = np.array(
        [
            [0, 0],
            [TABLE_W, 0],
            [TABLE_W, TABLE_H],
            [0, TABLE_H],
        ],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(src_points, dst_points)
    return H, dst_points


def get_zone(x, y):
    if x < MID_X and y < MID_Y:
        return 3
    elif x >= MID_X and y < MID_Y:
        return 1
    elif x < MID_X and y >= MID_Y:
        return 4
    else:
        return 2


class TableTennisProcessor:
    def __init__(self, model_path, corners_json, conf=0.2, iou=0.7):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

        self.src_corners = load_table_corners(corners_json)
        self.H, self.dst_points = compute_homography_matrix(self.src_corners)
        if self.H is None:
            raise RuntimeError("Homography matrix could not be computed")

        self.trajectories = {}
        self.top_view_trajectories = {}

        # Game logic
        self.ball_state = BallState()
        self.current_game = Game()
        self.match = Match(best_of=5)
        self.rally = RallyFSM()

    def process_frame(self, frame: np.ndarray):
        # ------------------------------
        # Рабочая копия кадра для OpenCV
        # ------------------------------
        frame = frame.copy()

        top_view = np.zeros((TABLE_H, TABLE_W, 3), dtype=np.uint8)
        cv2.line(top_view, (MID_X, 0), (MID_X, TABLE_H), (255, 255, 255), 2)
        cv2.line(top_view, (0, MID_Y), (TABLE_W, MID_Y), (255, 255, 255), 2)
        table_outline = np.array(self.dst_points, dtype=np.int32)
        cv2.polylines(top_view, [table_outline], True, (0, 255, 255), 3)

        # ------------------------------
        # YOLO detection
        # ------------------------------
        try:
            results = self.model(source=frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        except Exception:
            # Если YOLO упала, просто возвращаем frame без обработки
            return frame, top_view

        # ------------------------------
        # Draw table corners
        # ------------------------------
        for i, corner in enumerate(self.src_corners):
            cv2.circle(frame, tuple(corner.astype(int)), 8, (0, 255, 255), -1)
            cv2.putText(frame, str(i + 1), (int(corner[0]) + 10, int(corner[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        pts = self.src_corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        if results.boxes is None:
            return frame, top_view

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if cls == 0:
                color = BALL_COLOR
                label = f"Ball {conf:.2f}"
                track_id = 0

                self.trajectories.setdefault(track_id, []).append((cx, cy))
                # Храним только последние MAX_TRAJECTORY_POINTS
                self.trajectories[track_id] = self.trajectories[track_id][-MAX_TRAJECTORY_POINTS:]

                pts_tr = np.array(self.trajectories[track_id], np.int32)
                if len(pts_tr) > 1:
                    cv2.polylines(frame, [pts_tr], False, TRAJECTORY_COLOR, TRAJECTORY_THICKNESS)

                pt = np.array([[[cx, cy]]], dtype=np.float32)
                mx, my = cv2.perspectiveTransform(pt, self.H)[0][0].astype(int)

                self.ball_state.update(mx, my)
                event = detect_event(self.ball_state, mx, my)
                side = side_of_table(mx)
                loser = self.rally.step(event, side)
                if loser:
                    winner = LEFT if loser == RIGHT else RIGHT
                    self.current_game.add_point(winner)
                    if self.current_game.finished:
                        self.match.games_won[winner] += 1
                        self.rally.reset()
                        self.current_game = Game()

                self.top_view_trajectories.setdefault(track_id, []).append((mx, my))
                self.top_view_trajectories[track_id] = self.top_view_trajectories[track_id][-MAX_TOP_VIEW_POINTS:]
                cv2.circle(top_view, (mx, my), 10, BALL_COLOR, -1)
                pts_tv = np.array(self.top_view_trajectories[track_id], np.int32)
                if len(pts_tv) > 1:
                    cv2.polylines(top_view, [pts_tv], False, BALL_COLOR, 3)
                zone = get_zone(mx, my)
                cv2.putText(top_view, f"Z{zone}", (mx + 15, my - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, BALL_COLOR, 2)
            else:
                color = OTHER_COLOR
                label = f"Class {cls} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(top_view, f"{self.current_game.score[LEFT]} : {self.current_game.score[RIGHT]}",
                    (TABLE_W // 2 - 60, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return frame, top_view
