from django.urls import path
from api.views import predict

urlpatterns = [
    path('predict/', predict, name='predict'),
]

