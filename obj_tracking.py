import cv2
import numpy as np
from object_detection import ObjectDetection
import math

stream_url = "https://cctvn5.freeway.gov.tw/abs2mjpg/bmjpg?camera=f79f2f81-126d-450f-9152-9c844567a233"
# Load the video
cap = cv2.VideoCapture(stream_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 避免幀積壓
cap.set(cv2.CAP_PROP_FPS, 10)  # 限制FPS，降低負載
if not cap.isOpened():
    raise RuntimeError("Unable to open camera. Check device index and permissions.")

od = ObjectDetection()


class TrackableObject:
    def __init__(self, object_id, center_point, current_frame=0):
        self.id = object_id
        self.center_point: tuple[int, int] = center_point
        self.updated_frames = current_frame


SHOW_SLIDER_FOR_DEBUG = False


# int, TrackableObject
tracking_obj: dict[int, TrackableObject] = {}
auto_id = 0

frame_count = 0

retry_count = 0
# os time
start_time = cv2.getTickCount()

prev_count = 0

HEIGHT = 360
WIDTH = 720


prev_left_count = 0
prev_right_count = 0

if SHOW_SLIDER_FOR_DEBUG:
    cv2.startWindowThread()
    cv2.namedWindow("Slider", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Slider", 400, 100)
    cv2.createTrackbar("x1", "Slider", 0, WIDTH, lambda x: None)
    cv2.createTrackbar("y1", "Slider", 0, HEIGHT, lambda x: None)
    cv2.createTrackbar("x2", "Slider", WIDTH, WIDTH, lambda x: None)
    cv2.createTrackbar("y2", "Slider", HEIGHT, HEIGHT, lambda x: None)

debug_x1, debug_y1, debug_x2, debug_y2 = 35, 0, 435, HEIGHT


def get_column(x, y):
    # check if m < 0, below (>0) is right, otherwise left
    m = (debug_y2 - debug_y1) / (debug_x2 - debug_x1)
    dy = (m * x - debug_x1 * m) - y
    return 0 if (dy > 0) == (m < 0) else 1


while True:
    if SHOW_SLIDER_FOR_DEBUG:
        x1 = cv2.getTrackbarPos("x1", "Slider")
        y1 = cv2.getTrackbarPos("y1", "Slider")
        x2 = cv2.getTrackbarPos("x2", "Slider")
        y2 = cv2.getTrackbarPos("y2", "Slider")
        if x2 > x1:
            debug_x1, debug_x2 = x1, x2
        else:
            debug_x1, debug_x2 = 0, WIDTH
        if y2 > y1:
            debug_y1, debug_y2 = y1, y2
        else:
            debug_y1, debug_y2 = 0, HEIGHT
        print(f"Debug Line: ({debug_x1}, {debug_y1}) to ({debug_x2}, {debug_y2})")

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
            distance = math.hypot( (pt2[0] - pt1[0]), (pt2[1] - pt1[1]))
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
            tracking_obj[auto_id] = TrackableObject(auto_id, pt1, frame_count)
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
        else:
            prev_right_count += 1
        del tracking_obj[id]
    cv2.line(frame, (debug_x1, debug_y1), (debug_x2, debug_y2), (255, 0, 0), 2)
    print(
        f"Vehicle count: {len(tracking_obj)+prev_count} ({(len(tracking_obj)+prev_count)/duration:.2f} Vehicle/s),Left: {left_count+ prev_left_count}, Right: {right_count+prev_right_count}, FPS: {frame_count/duration:.2f}"
    )
    frame_to_show = cv2.resize(frame, (2 * WIDTH, 2 * HEIGHT))
    cv2.putText(
        frame_to_show,
        f"Vehicle count: {len(tracking_obj)+prev_count} ({(len(tracking_obj)+prev_count)/duration:.2f} Vehicle/s)",
        (10, 2 * HEIGHT - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame_to_show,
        f"Left: {left_count+ prev_left_count}, Right: {right_count+prev_right_count}",
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

    if cv2.waitKey(0 if SHOW_SLIDER_FOR_DEBUG else 1) & 0xFF == ord("q"):
        break
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
