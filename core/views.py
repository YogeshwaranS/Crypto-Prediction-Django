from django.shortcuts import render
from django.views import generic
import json
from Crypt.scripts import ScriptManager
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt


# Create your views here.
class IndexView(generic.ListView):
    template_name = 'core/index.html'

    def get(self, request, *args, **kwargs):
        # import_bonds()
        # first_item_draws_dic = get_draws_dic_for_bond_name(constants.AVAILABLE_DENOMINATIONS[1])
        # context = {'available_domination_name': constants.AVAILABLE_DENOMINATIONS,
        #            'first_item_draws_dic': first_item_draws_dic}
        # context['latest_draws_dic'] = latest_draws()
        graph_files_list = ScriptManager.get_graph_files()
        context = {"graph_files_list": graph_files_list}
        return render(self.request, template_name=self.template_name, context=context)


@csrf_exempt
def download_latest_data(request):
    ScriptManager.download_data()
    print ("download_latest_data_clicked")
    context = {
        'error': '1',
    }
    return HttpResponse(json.dumps(context))


@csrf_exempt
def generate_get_graph(request):
    import random
    import django
    import datetime

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt
    import io
    import os
    from Crypt.scripts import Step_4_CNN_PlotV1
    print ("generate_graph request")
    file_name = request.GET['file']
    fld = os.getcwd() + "/Crypt/scripts/" + file_name
    plot = Step_4_CNN_PlotV1.get_plot(fld)
    f = Figure()
    canvas = FigureCanvas(f)
    buf = io.BytesIO()
    plot.savefig(buf, format='png')
    plot.close(f)
    response = HttpResponse(buf.getvalue(), content_type='image/png')

    return render(request, "core/graph.html", {'file': response})


@csrf_exempt
def generate_graph(request):
    import random
    import django
    import datetime

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter
    import matplotlib.pyplot as plt
    import io
    import os
    from Crypt.scripts import Step_4_CNN_PlotV1
    print ("generate_graph request")
    file_name = request.POST['name']
    fld = os.getcwd() + "/Crypt/scripts/" + file_name
    plot = Step_4_CNN_PlotV1.get_plot(fld)
    f = Figure()
    canvas = FigureCanvas(f)
    buf = io.BytesIO()
    plot.savefig(buf, format='png')
    plot.close(f)
    # plot.show()
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response


@csrf_exempt
def generate_high_chart_graph(request):
    import os
    from Crypt.scripts import Step_4_CNN_PlotV1
    fld = os.getcwd() + "/Crypt/scripts/bitcoin2015to2017_close_CNN_2_relu-01-0.01066.hdf5"
    dic = Step_4_CNN_PlotV1.get_plot_df(fld)
    ground_true_df = dic['ground_true_df']
    prediction_df = dic['prediction_df']
    x_cat_list = []
    actual_cat_list = []
    actual_val_list = []
    prediction_cat_list = []
    prediction_val_list = []
    for item in ground_true_df.times:
        actual_cat_list.append(str(item).split(' ')[0])
    for item in prediction_df.times:
        prediction_cat_list.append(str(item))
    for item in ground_true_df.value:
        actual_val_list.append(str(item))
    for item in prediction_df.value:
        prediction_val_list.append(str(item))
    print (len(actual_cat_list), len(prediction_cat_list))
    costomePr = prediction_val_list[0:len(actual_val_list)]
    print (costomePr)
    print (actual_val_list)
    print (len(costomePr), len(actual_val_list))
    actual_val_list = list(map(int, actual_val_list))
    prediction_val_list = list(map(int, prediction_val_list))
    context = {
        'x-axis-labels': actual_cat_list,
        'actual-data': actual_val_list,
        'prediction-data': prediction_val_list,
    }
    return HttpResponse(json.dumps(context))
