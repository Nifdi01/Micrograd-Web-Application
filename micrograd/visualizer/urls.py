from django.urls import path
from . import views

app_name = 'visualizer'

urlpatterns = [
    path('', views.visualization, name='visualizer'),
]
