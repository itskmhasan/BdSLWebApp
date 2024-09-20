from django.urls import path

from detector import views

urlpatterns = [
    path('', views.upload_image, name='index'),
]
