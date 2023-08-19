from graphviz import Digraph
import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self.label = label
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1) / (math.exp(2*n)+1)
        out=Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1-out.data**2) * out.grad
        out._backward = _backward
        
        return out
    

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    

def trace(root):
    # Build the set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(node):
        if node not in nodes:
            nodes.add(node)
            for child in node._prev:
                edges.add((child, node))
                build(child)

    build(root)
    return nodes, edges
 
    
import pydot

def draw_dot(root):
    graph = pydot.Dot(graph_type='digraph', rankdir='LR')  # LR means Left-to-Right

    nodes, edges = trace(root)
    node_ids = {}  # A dictionary to store the node IDs

    for n in nodes:
        uid = str(id(n))
        # If value, create a rectangular node
        node_label = f'{{ {n.label} | data {n.data:.4f} | grad {n.grad:.4f} }}'
        node = pydot.Node(uid, label=node_label, shape='record')
        node.set("style", "filled")
        node.set("fontname", "Arial")
        node.set("fillcolor", "lightblue")
        graph.add_node(node)
        node_ids[n] = uid  # Store the mapping of node to ID

        if n._op:
            # If operation, create circular node
            op_node = pydot.Node(uid + n._op, label=n._op, shape='ellipse')
            op_node.set("style", "filled")
            op_node.set("fontname", "Roboto")
            op_node.set("bgcolor", "magenta")
            graph.add_node(op_node)
            graph.add_edge(pydot.Edge(uid + n._op, uid))

    for n1, n2 in edges:
        if n1 in node_ids and n2 in node_ids:
            graph.add_edge(pydot.Edge(node_ids[n1], node_ids[n2] + n2._op))  # Connect Nodes through the operation

    return graph.to_string()
