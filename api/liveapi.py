from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cv2
import math
import time
import threading
from typing import Generator

from model.traffic.vehicle_tracking import (
	get_traffic_data,
	TrafficData,
	_safe_divide,  # reuse internal helper
	get_default_stream_url,
)
from model.traffic.object_detection import ObjectDetection


class TrafficDataResponse(BaseModel):
	total_count: int
	rate: float
	left_count: int
	right_count: int
	left_rate: float
	right_rate: float
	duration: float
	avg_speed: float
	avg_left_speed: float
	avg_right_speed: float

	@classmethod
	def from_td(cls, td: TrafficData) -> "TrafficDataResponse":  # type: ignore[name-defined]
		return cls(**td.to_json())


app = FastAPI(title="Traffic Monitoring API", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
async def health():
	return {"status": "ok"}


@app.get("/traffic", response_model=TrafficDataResponse)
async def traffic(
	frames: int = Query(120, ge=1, le=1000, description="Maximum frames to analyze (higher = more accurate but slower)"),
	stream_url: Optional[str] = Query(
		None, description="Custom camera stream URL; if empty, use the default public freeway camera"
	),
):
	"""Run a bounded frame capture and return aggregated traffic metrics.

	Estimated time: approximately frames / (camera FPS). For example, if source is 10 FPS, frames=120 ≈ 12 seconds.
	"""
	# Move CPU-bound synchronous work to a thread to avoid blocking the event loop
	td: TrafficData = await asyncio.to_thread(
		get_traffic_data, max_frames=frames, stream_url=stream_url
	)
	return TrafficDataResponse.from_td(td)


# ===================== Real-time streaming tracker ===================== #
class RealTimeTrafficTracker:
	"""Continuous tracker running in a background thread producing annotated frames.

	Uses similar logic to `get_traffic_data` but does not stop after a fixed number of frames.
	It continuously updates counts and average speeds for the /stream video and /stats queries.
	"""

	def __init__(self, stream_url: str | None = None, width: int = 720, height: int = 360):
		self.stream_url = stream_url or get_default_stream_url()
		self.WIDTH = width
		self.HEIGHT = height
		self.capture: cv2.VideoCapture | None = None
		self.od = ObjectDetection()
		self.running = False
		self.thread: threading.Thread | None = None
		self.lock = threading.Lock()

		# tracking state
		self.tracking_obj: dict[int, dict] = {}
		self.auto_id = 0
		self.prev_count = 0
		self.prev_left_count = 0
		self.prev_right_count = 0
		self.left_speeds: list[float] = []
		self.right_speeds: list[float] = []
		self.start_time = time.time()
		self.last_frame_jpeg: bytes | None = None
		self.last_stats: dict | None = None
		self.ROAD_DISTANCE = {"left": 50, "right": 50}
		self.central_line_calculated = False
		self.x1 = 0
		self.y1 = 0
		self.x2 = self.WIDTH
		self.y2 = self.HEIGHT
		self.retry_count = 0

	def _open_capture(self):
		if self.capture is not None:
			self.capture.release()
		self.capture = cv2.VideoCapture(self.stream_url)
		self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
		self.capture.set(cv2.CAP_PROP_FPS, 10)
		if not self.capture.isOpened():
			raise RuntimeError("Unable to open camera stream for real-time tracker")

	def start(self):
		if self.running:
			return
		self.running = True
		self.thread = threading.Thread(target=self._loop, daemon=True)
		self.thread.start()

	def stop(self):
		self.running = False
		if self.thread and self.thread.is_alive():
			self.thread.join(timeout=2)
		if self.capture:
			self.capture.release()

	def get_column(self, x: int, y: int) -> int:
		m = (self.y2 - self.y1) / (self.x2 - self.x1) if (self.x2 - self.x1) != 0 else 0.00001
		dy = (m * x - self.x1 * m + self.y1) - y
		return 0 if (dy > 0) == (m < 0) else 1

	def _update_central_line(self, frame):
		if self.central_line_calculated:
			return
		edge = cv2.Canny(frame, 100, 200)
		edge = cv2.dilate(edge, None, iterations=2)
		edge = cv2.erode(edge, None, iterations=3)
		edge = cv2.GaussianBlur(edge, (5, 5), 0)
		lines = cv2.HoughLinesP(edge, 1, math.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
		if lines is not None:
			tavgx1 = tavgy1 = tavgx2 = tavgy2 = 0
			count = 0
			for line in lines:
				tx1, ty1, tx2, ty2 = line[0]
				if (abs(ty2 - ty1) > 0.5 * abs(tx2 - tx1)) and abs((tx1 + tx2) / 2 - (self.WIDTH / 2 - 50)) < 100 and abs((ty1 + ty2) / 2 - self.HEIGHT / 2) < 100:
					count += 1
					tavgx1 += tx1
					tavgy1 += ty1
					tavgx2 += tx2
					tavgy2 += ty2
			if count > 0:
				self.x1 = tavgx1 // count
				self.y1 = tavgy1 // count
				self.x2 = tavgx2 // count
				self.y2 = tavgy2 // count
				self.central_line_calculated = True

	def _loop(self):
		try:
			self._open_capture()
		except Exception as e:
			print("[RealTimeTrafficTracker] Initial open failed:", e)
			self.running = False
			return
		while self.running:
			if not self.capture:
				break
			ret, frame = self.capture.read()
			if not ret:
				# retry reconnect
				self.retry_count += 1
				if self.retry_count > 5:
					print("[RealTimeTrafficTracker] Too many read failures, stopping.")
					break
				time.sleep(0.5)
				try:
					self._open_capture()
				except Exception as e:
					print("[RealTimeTrafficTracker] Re-open failed:", e)
				continue
			self.retry_count = 0
			frame = cv2.resize(frame, (self.WIDTH, self.HEIGHT))
			self._update_central_line(frame)
			(class_ids, scores, boxes) = self.od.detect(frame)
			frame_count_duration = time.time() - self.start_time

			# tracking update (simplified from original)
			center_points_curr: list[tuple[int, int]] = []
			for box in boxes:
				(x, y, w, h) = box
				cx, cy = int(x + w / 2), int(y + h / 2)
				if w > 0.7 * self.WIDTH or h > 0.7 * self.HEIGHT or w < 60 or h < 60:
					continue
				center_points_curr.append((cx, cy))
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

			# match existing ids
			for pt1 in center_points_curr:
				best_id = None
				best_dist = float("inf")
				for oid, obj in self.tracking_obj.items():
					pt2 = obj["center"]
					distance = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
					if distance < 200 and distance < best_dist and (frame_count_duration - obj["last_time"]) < 2:
						best_id = oid
						best_dist = distance
				if best_id is not None:
					self.tracking_obj[best_id]["center"] = pt1
					self.tracking_obj[best_id]["last_time"] = frame_count_duration
				else:
					self.tracking_obj[self.auto_id] = {
						"center": pt1,
						"start_time": frame_count_duration,
						"last_time": frame_count_duration,
					}
					self.auto_id += 1

			# determine lost objects
			to_delete = []
			for oid, obj in self.tracking_obj.items():
				if frame_count_duration - obj["last_time"] > 1.5:  # lost
					to_delete.append(oid)
			for oid in to_delete:
				o = self.tracking_obj[oid]
				is_left = self.get_column(o["center"][0], o["center"][1]) == 0
				self.prev_count += 1
				duration_obj = frame_count_duration - o["start_time"]
				if duration_obj > 0:
					if is_left:
						self.prev_left_count += 1
						self.left_speeds.append(_safe_divide(self.ROAD_DISTANCE["left"], duration_obj))
					else:
						self.prev_right_count += 1
						self.right_speeds.append(_safe_divide(self.ROAD_DISTANCE["right"], duration_obj))
				del self.tracking_obj[oid]

			# annotate current ids
			for oid, obj in self.tracking_obj.items():
				is_left = self.get_column(obj["center"][0], obj["center"][1]) == 0
				cv2.putText(
					frame,
					str(oid),
					obj["center"],
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(0, 255, 0) if is_left else (255, 0, 0),
					2,
				)
			cv2.line(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2)

			# compute stats snapshot
			left_speed_kmh = _safe_divide(sum(self.left_speeds), len(self.left_speeds)) * 3.6
			right_speed_kmh = _safe_divide(sum(self.right_speeds), len(self.right_speeds)) * 3.6
			all_speeds = self.left_speeds + self.right_speeds
			avg_speed_kmh = _safe_divide(sum(all_speeds), len(all_speeds)) * 3.6
			duration_total = frame_count_duration
			total_count = self.prev_count

			overlay_lines = [
				f"Vehicles: {total_count} ({_safe_divide(total_count, duration_total):.2f}/s)",
				f"Left: {self.prev_left_count} Right: {self.prev_right_count}",
				f"Speed L: {left_speed_kmh:.1f} R: {right_speed_kmh:.1f} Avg: {avg_speed_kmh:.1f} (relative)",
			]
			for i, text in enumerate(overlay_lines):
				cv2.putText(
					frame,
					text,
					(10, self.HEIGHT - 10 - (len(overlay_lines) - 1 - i) * 22),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.6,
					(255, 255, 255),
					2,
				)

			# encode frame
			ret_jpg, jpg = cv2.imencode('.jpg', frame)
			if ret_jpg:
				with self.lock:
					self.last_frame_jpeg = jpg.tobytes()
					self.last_stats = {
						"total_count": total_count,
						"rate": _safe_divide(total_count, duration_total),
						"left_count": self.prev_left_count,
						"right_count": self.prev_right_count,
						"left_rate": _safe_divide(self.prev_left_count, duration_total),
						"right_rate": _safe_divide(self.prev_right_count, duration_total),
						"duration": duration_total,
						"avg_speed": avg_speed_kmh,
						"avg_left_speed": left_speed_kmh,
						"avg_right_speed": right_speed_kmh,
					}
			# small sleep to reduce CPU if needed
			time.sleep(0.01)
		self.running = False

	def frame_generator(self) -> Generator[bytes, None, None]:
		boundary = b"--frame"
		while self.running:
			with self.lock:
				frame = self.last_frame_jpeg
			if frame is None:
				time.sleep(0.05)
				continue
			yield boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
			time.sleep(0.05)  # ~20 fps upper bound

	def get_stats(self):
		with self.lock:
			return self.last_stats.copy() if self.last_stats else None


tracker = RealTimeTrafficTracker()


@app.on_event("startup")
async def _startup():
	try:
		tracker.start()
	except Exception as e:
		print("[startup] Failed to start tracker:", e)


@app.on_event("shutdown")
async def _shutdown():
	tracker.stop()


@app.get("/stream")
async def stream():
	if not tracker.running:
		raise HTTPException(status_code=503, detail="Tracker not running")

	gen = tracker.frame_generator()
	return StreamingResponse(gen, media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/stats")
async def stats():
	data = tracker.get_stats()
	if data is None:
		return JSONResponse({"status": "initializing"})
	return data


@app.get("/")
async def root():
	return {
		"message": "Traffic Monitoring API",
		"endpoints": ["/traffic", "/health", "/stream", "/stats"],
		"example_batch": "/traffic?frames=80",
		"example_stream": "/stream",
	}


if __name__ == "__main__":
	# Allow running directly with `python api/liveapi.py` (convenient for local testing)
	import uvicorn

	uvicorn.run("api.liveapi:app", host="0.0.0.0", port=8000, reload=True)

