import cv2 as cv
import json
import potrace
import sys
from pathlib import Path
from bitmap import mkbitmap


# [TODO] Parse input arguments
fin = Path('input.png')
fout = Path('calc.js')
ftemp = Path('templates') / 'image.json'

try:
    data = cv.imread(str(fin))
    assert data is not None
except (SystemError, AssertionError):
    print("Cannot open", fin)
    sys.exit()


# Trace curves on input data
SCALE = 2
height, width = data.shape[:2]
bitmap = mkbitmap(data, f=None, s=SCALE, t=0.5) == 0
points = [list() for _ in range(7)]
def append(x, point):
    points[2*x].append(point[0] / SCALE)
    points[2*x+1].append(height - point[1] / SCALE)
def copy(x, y, first=False):
    points[2*y].append(points[2*x][0 if first else -1])
    points[2*y+1].append(points[2*x+1][0 if first else -1])
def opaque(val):
    points[-1].append(val)
for curve in potrace.Bitmap(bitmap).trace():
    if points[0]:
        copy(0, 1)
        append(2, curve.start_point)
        opaque(False)
    append(0, curve.start_point)
    for segment in curve:
        if segment.is_corner:
            # TODO: line coalescence
            copy(0, 1)
            append(2, segment.c)
            opaque(True)
            append(0, segment.c)
            append(1, segment.c)
            append(2, segment.end_point)
        else:
            append(1, segment.c1)
            append(2, segment.c2)
        opaque(True)
        append(0, segment.end_point)
copy(0, 1)
copy(0, 2, first=True)
opaque(False)


# Write to a Desmos calculator
with open(ftemp, 'r') as template:
    state = json.load(template)

state['graph']['viewport']['xmax'] = width
state['graph']['viewport']['ymax'] = height
def num2str(n):
    return str(round(n, 2)).rstrip('0').rstrip('.')
table = next(filter( \
    lambda exp: exp['id'] == 'data' and exp['type'] == 'table', \
    state['expressions']['list']))
for column, values in zip(table['columns'], points):
    column['values'] = [num2str(num) for num in values]
    
with open(fout, 'w') as output:
    output.write(f"Calc.setState({json.dumps(state, separators=(',', ':'))})")
