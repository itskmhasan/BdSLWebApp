from django.urls import path

from detector import views, videos

urlpatterns = [
    path('', views.upload_image, name='index'),
    path('video_feed/', videos.video_feed, name='video_feed'),
]
