from graphviz import Digraph

dot = Digraph('SimpleCNN_vertical', format='png')
dot.engine = 'dot'

# Graph-level attributes: top→bottom, ảnh dọc
dot.graph_attr.update(
    rankdir='TB',      # TB = Top to Bottom
    splines='ortho',   # cạnh vuông góc
    nodesep='0.6',     # khoảng cách ngang giữa các node
    ranksep='1',       # khoảng cách dọc giữa các hàng
    dpi='300',
    size='6,12'        # width 6", height 12"
)

# Node-level styling
dot.node_attr.update(
    shape='box',
    style='rounded,filled',
    fillcolor='#AED6F1',
    fontname='Helvetica',
    fontsize='11',
    margin='0.2,0.1'
)

# Edge-level styling
dot.edge_attr.update(
    arrowhead='vee',
    arrowsize='0.7'
)

# Định nghĩa các lớp (node)
dot.node('I',   'Input\n1×28×28')
dot.node('C1',  'Conv2d\n1→32, 3×3')
dot.node('R1',  'ReLU')
dot.node('C2',  'Conv2d\n32→64, 3×3')
dot.node('R2',  'ReLU')
dot.node('P',   'MaxPool2d\n2×2')
dot.node('F',   'Flatten')
dot.node('FC1', 'Linear\n(64×12×12→128)')
dot.node('R3',  'ReLU')
dot.node('FC2','Linear\n(128→10)')
dot.node('O',   'Output\n1×10 logits')

# Đường nối
edges = [
    ('I',   'C1'),
    ('C1',  'R1'),
    ('R1',  'C2'),
    ('C2',  'R2'),
    ('R2',  'P'),
    ('P',   'F'),
    ('F',   'FC1'),
    ('FC1', 'R3'),
    ('R3',  'FC2'),
    ('FC2', 'O'),
]
for src, dst in edges:
    dot.edge(src, dst)

# Sinh file vertical PNG
dot.render('simple_cnn_forward_vertical', cleanup=True)
print("Đã lưu sơ đồ forward dọc ở simple_cnn_forward_vertical.png")
