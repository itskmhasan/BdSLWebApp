from django.urls import path

from detector import views, videos, tests

urlpatterns = [
    path('', views.upload_image, name='index'),
    path('test/', tests.upload_image, name='index'),
    path('video_feed/', videos.video_feed, name='video_feed'),
]
