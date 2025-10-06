from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/traffic/latest/', views.traffic_latest, name='traffic-latest'),
    path('api/traffic/all/', views.traffic_all, name='traffic-all'),
    path('api/video/url/', views.video_url, name='video-url'),
]
