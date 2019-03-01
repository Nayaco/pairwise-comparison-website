from django.urls import path

from . import views

app_name = 'polls'
urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('initvote/', views.initvote, name='initvote'),
    path('<int:pk>/', views.detail_getpair, name='detail'),#views.DetailView.as_view()
    path('<int:pk>/results/', views.ResultsView.as_view(), name='results'),
    path('<int:question_id>/vote/', views.vote, name='vote'),
    path('<int:question_id>/getpair/', views.getpair, name='getpair'),
    path('<int:question_id>/results/recalresult/', views.recalresult, name='recalresult'),
    path('<int:question_id>/results/getparam/', views.getparam, name='getparam'),
]
