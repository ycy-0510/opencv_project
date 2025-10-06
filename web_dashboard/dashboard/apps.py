from django.apps import AppConfig
import threading
import time
import logging

from django.conf import settings

logger = logging.getLogger(__name__)


# 簡單的記憶體暫存（若要長期保留請改用 DB / cache）

LATEST_TRAFFIC_DATA = {  # 最新一次（含 payload）
    "ts": None,
    "payload": None,  # 扁平化欄位字典
    "error": None,
}

TRAFFIC_DATA = []  # 歷史扁平化資料列，每筆: {ts, total_count, left_count, ...}


class DashboardConfig(AppConfig):
    default_auto_field = "django.db.models.AutoField"
    name = "dashboard"

    _started = False  # 防止 runserver autoreload 啟動兩次

    def ready(self):
        # 避免 migrations 或 shell 之類流程意外啟動背景排程，可加條件
        if getattr(self.__class__, "_started", False):
            return
        self.__class__._started = True

        interval = getattr(settings, "TRAFFIC_JOB_INTERVAL", 600)  # seconds (預設 10 分鐘)
        history_limit = getattr(settings, "TRAFFIC_HISTORY_LIMIT", 1000)

        def worker_loop():
            from django.utils import timezone
            from model.traffic.vehicle_tracking import get_traffic_data

            logger.info("Traffic background worker started, interval=%s", interval)
            while True:
                start_at = time.time()
                try:
                    data = get_traffic_data()
                    ts = timezone.now().isoformat()
                    # 更新全域最新資料（不重新指派變數，避免遮蔽）
                    LATEST_TRAFFIC_DATA["ts"] = ts
                    LATEST_TRAFFIC_DATA["payload"] = {
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
                    LATEST_TRAFFIC_DATA["error"] = None
                    logger.info("Traffic snapshot updated: %s", LATEST_TRAFFIC_DATA["payload"])
                    # 扁平化歷史資料列
                    flat_entry = {"ts": ts, **LATEST_TRAFFIC_DATA["payload"]}
                    TRAFFIC_DATA.append(flat_entry)
                    if len(TRAFFIC_DATA) > history_limit:
                        # 移除最舊資料
                        del TRAFFIC_DATA[0: len(TRAFFIC_DATA) - history_limit]
                except Exception as e:  # pylint: disable=broad-except
                    LATEST_TRAFFIC_DATA["error"] = str(e)
                    logger.exception("Traffic job failed: %s", e)
                # 計算剩餘睡眠時間，避免長執行造成偏移太多
                elapsed = time.time() - start_at
                sleep_time = max(5, interval - elapsed)
                time.sleep(sleep_time)

        t = threading.Thread(target=worker_loop, name="traffic-worker", daemon=True)
        t.start()
        logger.info("DashboardConfig.ready() started background traffic worker thread")
