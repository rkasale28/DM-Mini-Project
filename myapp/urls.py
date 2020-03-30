from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.home, name='home'),
    path('naive',views.naive_bayes,name='naive_bayes'),
    path('knn',views.knn,name='knn'),
    path('slr',views.slr,name='slr'),
    path('sample',views.sample,name='sample')
]
