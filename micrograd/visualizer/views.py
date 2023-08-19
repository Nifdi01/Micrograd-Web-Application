from django.shortcuts import render
# from grad_module import nn
from grad_module.engine import Value, draw_dot
import pydot
import base64
from django.http import JsonResponse


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


def generate_graph_from_code(user_code):
    # Create a custom namespace for execution
    code_namespace = {'Value': Value, 'tanh': Value.tanh, 'relu': Value.relu}

    try:
        # Execute the user's code
        exec(user_code, code_namespace)
    except Exception as e:
        return None, str(e)  # Return None and error message on code execution error

    # Extract the computation graph from the namespace
    if 'L' in code_namespace:
        root = code_namespace['L']
        root.backward()
        gradient_dict = {}
        
        for var_name, var_node in code_namespace.items():
            if isinstance(var_node, Value):
                gradient_dict[var_name] = var_node.grad   
        # Add custom operations like tanh and relu to the computation graph
        for name, operation in code_namespace.items():
            if callable(operation) and hasattr(operation, '_op'):
                # Assuming the operations have an attribute '_op' indicating their name
                root._prev.add(operation)
        
        dot_string = draw_dot(root)
        return dot_string, gradient_dict
    else:
        return None, "No computation graph defined"


def visualization(request):
    if request.method == 'POST':
        user_code = request.POST.get('user_code')
        dot_string, error_message = generate_graph_from_code(user_code)

        if dot_string is not None:
            graph = pydot.graph_from_dot_data(dot_string)
            image_data = graph[0].create_png()

            image_base64 = base64.b64encode(image_data).decode('utf-8')

            return JsonResponse({'image_base64': image_base64})
        else:
            return JsonResponse({'error': error_message})

    dot_string = generate_dot_graph()

    graph = pydot.graph_from_dot_data(dot_string)
    image_data = graph[0].create_png()

    image_base64 = base64.b64encode(image_data).decode('utf-8')

    return render(request, 'visualizer/base.html', {'image_base64': image_base64})
