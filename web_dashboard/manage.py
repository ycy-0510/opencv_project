#!/usr/bin/env python
import os
import sys
from pathlib import Path


def main():
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'dashboard_project.settings')
    # 確保可以匯入到專案根目錄下的 model/ 等模組
    base_dir = Path(__file__).resolve().parent
    parent = base_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and available on your PYTHONPATH?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
