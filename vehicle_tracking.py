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
    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
