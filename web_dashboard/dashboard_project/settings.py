"""Django settings for dashboard_project.

Minimal settings for a local dashboard inside the workspace.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'dev-secret-key'

DEBUG = True

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.staticfiles',
    'dashboard',
]

MIDDLEWARE = [
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'dashboard_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
            ],
        },
    },
]

WSGI_APPLICATION = 'dashboard_project.wsgi.application'

# Static files (CSS, JavaScript, Images)
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

# 未使用媒體檔案設定（保留為未來擴充空間）
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / '_unused_media'

DEFAULT_AUTO_FIELD = 'django.db.models.AutoField'
