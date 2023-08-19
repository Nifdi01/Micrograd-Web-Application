from django.shortcuts import render
# from grad_module import nn
from grad_module.engine import Value, draw_dot
import pydot
import base64

def generate_dot_graph():
    # Inputs
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')

    # Weights
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')

    # Bias
    b = Value(8.0, label='b')

    # x1w1 + x2w2 + b
    x1w1 = x1 * w1
    x1w1.label = 'x1w1'
    x2w2 = x2 * w2
    x2w2.label = 'x2w2'
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = 'x1w1+x2w2'
    n = x1w1x2w2 + b
    n.label = 'n'
    o = n.tanh()
    o.label='o'
    o.backward()
    dot_string = draw_dot(o)
    return dot_string

def visualization(request):
    dot_string = generate_dot_graph()

    graph = pydot.graph_from_dot_data(dot_string)
    image_data = graph[0].create_png()

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    return render(request, 'visualizer/base.html', {'image_base64': image_base64})
