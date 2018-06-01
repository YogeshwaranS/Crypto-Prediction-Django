from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^download_latest_data/$', views.download_latest_data, name='download_latest_data'),
    # url(r'generate_graph.png', views.generate_graph, name='generate_graph'),
    # url(r'^generate_graph.png$', views.generate_graph)
    url(r'^generate_get_graph/$', views.generate_get_graph, name='generate_get_graph'),
    url(r'^generate_graph.png$', views.generate_graph, name='generate_graph'),
    url(r'^generate_high_chart_graph/$', views.generate_high_chart_graph, name='generate_high_chart_graph'),
]
