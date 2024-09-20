from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

from BdSLWebApp import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detector.urls')),
]

# Serve media files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
