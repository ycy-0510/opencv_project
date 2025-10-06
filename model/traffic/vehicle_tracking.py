import cv2
import numpy as np
from model.traffic.object_detection import ObjectDetection
import math


class TrackableObject:
    def __init__(self, object_id, center_point, current_frame=0, start_time=0.0):
        self.id = object_id
        self.center_point: tuple[int, int] = center_point
        self.updated_frames = current_frame
        self.startTime = start_time


class TrafficData:
    def __init__(
        self,
        total_count: int,
        rate: float,
        left_count=int,
        right_count=int,
        left_rate=float,
        right_rate=float,
        duration=float,
        avg_speed=float,
        avg_left_speed=float,
        avg_right_speed=float,
    ):
        self.total_count = total_count
        self.rate = rate  # vehicles per second
        self.left_count = left_count
        self.right_count = right_count
        self.left_rate = left_rate
        self.right_rate = right_rate
        self.duration = duration
        self.avg_speed = avg_speed # in km/h
        self.avg_left_speed = avg_left_speed
        self.avg_right_speed = avg_right_speed

    def to_string(self):
        return f"Total vehicles: {self.total_count}, Rate: {self.rate:.2f} vehicles/s, Left vehicles: {self.left_count}, Left Rate: {self.left_rate:.2f} vehicles/s, Right vehicles: {self.right_count}, Right Rate: {self.right_rate:.2f} vehicles/s, Duration: {self.duration:.2f} s, Average Speed: {self.avg_speed:.2f} km/h, Average Left Speed: {self.avg_left_speed:.2f} km/h, Average Right Speed: {self.avg_right_speed:.2f} km/h"


def _safe_divide(a, b):
    return a / b if b != 0 else 0.0


def get_traffic_data():

    stream_url = "https://cctvn5.freeway.gov.tw/abs2mjpg/bmjpg?camera=f79f2f81-126d-450f-9152-9c844567a233"
    # Load the video
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 避免幀積壓
    cap.set(cv2.CAP_PROP_FPS, 10)  # 限制FPS，降低負載
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera. Check device index and permissions.")

    od = ObjectDetection()

    # int, TrackableObject
    tracking_obj: dict[int, TrackableObject] = {}
    auto_id = 0

    frame_count = 0

    retry_count = 0
    # os time
    start_time = cv2.getTickCount()

    prev_count = 0

    ROAD_DISTANCE = {
        "left": 32,
        "right": 32,
    }  # meters, approximate distance of the road segment being monitored

    HEIGHT = 360
    WIDTH = 720

    prev_left_count = 0
    prev_right_count = 0

    left_speeds = []
    right_speeds = []

    x1, y1, x2, y2 = 0, 0, WIDTH, HEIGHT
    central_line_calculated = False

    def get_column(x, y):
        # check if m < 0, below (>0) is right, otherwise left
        m = (y2 - y1) / (x2 - x1)
        dy = (m * x - x1 * m + y1) - y
        return 0 if (dy > 0) == (m < 0) else 1

    while True:
        duration = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        ret, frame = cap.read()
        if not ret:
            if retry_count < 5:
                retry_count += 1
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue
            break
        retry_count = 0

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        if not central_line_calculated:
            edge = cv2.Canny(frame, 100, 200)
            # strong the line of edge
            edge = cv2.dilate(edge, None, iterations=2)
            # thin the line of edge
            edge = cv2.erode(edge, None, iterations=3)
            edge = cv2.GaussianBlur(edge, (5, 5), 0)
            # find the central line of edge
            lines = cv2.HoughLinesP(
                edge, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
            )
            if lines is not None:
                tavgx1, tavgy1, tavgx2, tavgy2 = 0, 0, 0, 0
                count = 0
                for line in lines:
                    tx1, ty1, tx2, ty2 = line[0]
                    if (
                        (abs(ty2 - ty1) > 0.5 * abs(tx2 - tx1))
                        and abs((tx1 + tx2) / 2 - (WIDTH / 2 - 50)) < 100
                        and abs((ty1 + ty2) / 2 - HEIGHT / 2) < 100
                    ):  # only keep near vertical lines
                        cv2.line(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                        count += 1
                        tavgx1 += tx1
                        tavgy1 += ty1
                        tavgx2 += tx2
                        tavgy2 += ty2
                if count > 0:
                    tavgx1 //= count
                    tavgy1 //= count
                    tavgx2 //= count
                    tavgy2 //= count
                    x1, y1, x2, y2 = tavgx1, tavgy1, tavgx2, tavgy2
                    central_line_calculated = True
        (class_ids, scores, boxes) = od.detect(frame)
        center_points_curr: list[tuple[int, int]] = []
        for box in boxes:
            (x, y, w, h) = box
            cx, cy = int(x + w / 2), int(y + h / 2)
            # cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            if w > 0.7 * WIDTH or h > 0.7 * HEIGHT or w < 60 or h < 60:
                continue
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            center_points_curr.append((cx, cy))
        for pt1 in center_points_curr:
            best_match_id = None
            best_match_distance = float("inf")
            for id, obj in tracking_obj.items():
                pt2 = obj.center_point
                distance = math.hypot((pt2[0] - pt1[0]), (pt2[1] - pt1[1]))
                if (
                    distance < 200
                    and distance < best_match_distance
                    and frame_count - obj.updated_frames == 1
                ):
                    best_match_id = id
                    best_match_distance = distance
            if best_match_id is not None:
                tracking_obj[best_match_id].center_point = pt1
                tracking_obj[best_match_id].updated_frames = frame_count
            else:
                tracking_obj[auto_id] = TrackableObject(
                    auto_id, pt1, frame_count, duration
                )
                auto_id += 1
        left_count = 0
        right_count = 0
        tracking_obj_to_delete = []
        for id, obj in tracking_obj.items():
            if frame_count - obj.updated_frames > 1:
                tracking_obj_to_delete.append(id)
                continue
            isLeft = get_column(obj.center_point[0], obj.center_point[1]) == 0
            if isLeft:
                left_count += 1
            else:
                right_count += 1
            cv2.putText(
                frame,
                str(id),
                obj.center_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0) if isLeft else (255, 0, 0),
                2,
            )
        for id in tracking_obj_to_delete:
            prev_count += 1
            isLeft = (
                get_column(
                    tracking_obj[id].center_point[0], tracking_obj[id].center_point[1]
                )
                == 0
            )
            if isLeft:
                prev_left_count += 1
                left_speeds.append(
                    _safe_divide(ROAD_DISTANCE["left"], (duration - tracking_obj[id].startTime))
                )
            else:
                prev_right_count += 1
                right_speeds.append(
                    _safe_divide(ROAD_DISTANCE["right"], (duration - tracking_obj[id].startTime))
                )
            del tracking_obj[id]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        print(
            f"Vehicle count: {len(tracking_obj)+prev_count} ({(len(tracking_obj)+prev_count)/duration:.2f} Vehicle/s),Left: {left_count+ prev_left_count}, Right: {right_count+prev_right_count}, FPS: {frame_count/duration:.2f}"
        )
        if __name__ == "__main__":
            frame_to_show = cv2.resize(frame, (2 * WIDTH, 2 * HEIGHT))
            cv2.putText(
                frame_to_show,
                f"Vehicle count: {len(tracking_obj)+prev_count} ({(len(tracking_obj)+prev_count)/duration:.2f} Vehicle/s)",
                (10, 2 * HEIGHT - 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_to_show,
                f"Left: {left_count+ prev_left_count}, Right: {right_count+prev_right_count}",
                (10, 2 * HEIGHT - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_to_show,
                f"Left Speed: {_safe_divide(sum(left_speeds), len(left_speeds))*3.6:.2f} km/h, Right Speed: {_safe_divide(sum(right_speeds), len(right_speeds))*3.6:.2f} km/h",
                (10, 2 * HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame_to_show,
                f"FPS: {frame_count/duration:.2f}",
                (10, 2 * HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Frame", frame_to_show)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        frame_count += 1
        if __name__ != "__main__":
            if frame_count >= 200:
                # clear current tracking and save to prev_count
                left_need_speed_data = len(left_speeds) < 3
                right_need_speed_data = len(right_speeds) < 3
                for id in tracking_obj.keys():
                    prev_count += 1
                    isLeft = (
                        get_column(
                            tracking_obj[id].center_point[0],
                            tracking_obj[id].center_point[1],
                        )
                        == 0
                    )
                    if isLeft:
                        prev_left_count += 1
                        if left_need_speed_data:
                            left_speeds.append(
                                _safe_divide(
                                    ROAD_DISTANCE["left"],
                                    (duration - tracking_obj[id].startTime),
                                )
                            )
                    else:
                        prev_right_count += 1
                        if right_need_speed_data:
                            right_speeds.append(
                                _safe_divide(
                                    ROAD_DISTANCE["right"],
                                    (duration - tracking_obj[id].startTime),
                                )
                            )
                break
    cap.release()
    cv2.destroyAllWindows()
    left_speeds = list(filter(lambda x: x>0 and x<110/3.6, left_speeds))
    right_speeds = list(filter(lambda x: x>0 and x<110/3.6, right_speeds))
    return TrafficData(
        total_count=prev_count,
        rate=prev_count / duration if duration > 0 else 0.0,
        left_count=prev_left_count,
        right_count=prev_right_count,
        left_rate=prev_left_count / duration if duration > 0 else 0.0,
        right_rate=prev_right_count / duration if duration > 0 else 0.0,
        duration=duration,
        avg_speed=_safe_divide(sum(left_speeds + right_speeds), len(left_speeds + right_speeds))*3.6,
        avg_left_speed=_safe_divide(sum(left_speeds), len(left_speeds))*3.6,
        avg_right_speed=_safe_divide(sum(right_speeds), len(right_speeds))*3.6,
    )


if __name__ == "__main__":
    get_traffic_data()
