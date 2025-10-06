from django.shortcuts import render
from django.http import JsonResponse

from .apps import LATEST_TRAFFIC_DATA, TRAFFIC_DATA
from model.traffic import vehicle_tracking as vt  # 使用專案根目錄中的 model 模組


def index(request):
    """Return the static default dashboard page (no dynamic data)."""
    return render(request, "dashboard/index.html")


def video_url(request):
    """Return the video stream URL used in the model.

    回傳格式：
    {
        "url": 影片串流 URL 字串
    }
    """
    return JsonResponse(
        {
            "url": vt.get_default_stream_url(),
        }
    )


def traffic_latest(request):
    """Return latest cached traffic data collected by background thread.

    回傳格式：
    {
        "ts": ISO時間或None,
        "data": {...} | None,
        "error": 錯誤訊息或 None
    }
    若目前尚未有資料，data 會是 None。
    如果你想即時重新抓一次，可改成 query 參數 trigger。
    """
    # 如果想支援強制重新抓，可加: if request.GET.get("refresh") == "1": ...
    payload = LATEST_TRAFFIC_DATA["payload"]
    if payload is None and request.GET.get("fallback_run") == "1":
        # 緊急同步跑一次（阻塞請求）。適度使用，避免阻塞過久。
        data = vt.get_traffic_data()
        payload = {
            "total_count": data.total_count,
            "rate": data.rate,
            "left_count": data.left_count,
            "right_count": data.right_count,
            "left_rate": data.left_rate,
            "right_rate": data.right_rate,
            "duration": data.duration,
            "avg_speed": data.avg_speed,
            "avg_left_speed": data.avg_left_speed,
            "avg_right_speed": data.avg_right_speed,
        }
    return JsonResponse(
        {
            "ts": LATEST_TRAFFIC_DATA["ts"],
            "data": payload,
            "error": LATEST_TRAFFIC_DATA["error"],
        }
    )

def traffic_all(request):
    """Return all cached traffic data collected by background thread.

    回傳格式：
    {
        "data": [ {...}, {...}, ... ]
    }
    若目前尚未有資料，data 會是空陣列。
    """
    return JsonResponse(
        {
            "data": TRAFFIC_DATA,
        }
    )