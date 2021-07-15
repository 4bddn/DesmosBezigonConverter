import argparse
import cv2 as cv
import json
import potrace
import sys
from math import isinf
from pathlib import Path

from bitmap import mkbitmap


# Parse input arguments
class Range(object):
    def __init__(self, start=None, end=None):
        self.start = start
        self.end = end
    def __eq__(self, num):
        return not isinf(num) and \
            (self.start is None or self.start <= num) and \
            (self.end is None or num <= self.end)
    def __repr__(self):
        return ', '.join(( \
            "(-∞" if self.start is None else f"[{self.start}", \
            "+∞)" if self.end is None else f"{self.end}]"))

parser = argparse.ArgumentParser(description="Trace a graphic and convert to a Desmos graph")
parser.add_argument('input', type=Path, help="input file containing an image", metavar='filename')
parser.add_argument('-o', '--output', help="name of the produced graph script", metavar='filename')
parser.add_argument('-f', '--filter', default=None, type=float, choices=[Range(0)], \
    help="use a high-pass filter with standard deviation n", metavar='n')
parser.add_argument('-t', '--threshold', default=0.5, type=float, choices=[Range(0, 1)], \
    help="brightness value for binary thresholding", metavar='n')
parser.add_argument('-ts', '--turdsize', default=2, type=float, choices=[Range(0)], \
    help="despeckle by ignoring areas smaller than a", metavar='a')
parser.add_argument('-am', '--alphamax', default=1.0, type=float, choices=[Range(0, 4/3)], \
    help="smoothness of the created curve", metavar='α')
parser.add_argument('-ot', '--opttolerance', default=0.2, type=float, choices=[Range(0)], \
    help="amount of error allowed in the tracing step", metavar='e')
args = parser.parse_args()
if args.output is None:
    args.output = args.input.with_suffix('.js')
args.template = Path('templates') / 'image.json'
args.scale = 2

try:
    data = cv.imread(str(args.input))
    assert data is not None
except (SystemError, AssertionError):
    print("Cannot open", args.input)
    sys.exit()


# Trace curves on input data
height, width = data.shape[:2]
bitmap = mkbitmap(data, f=args.filter, s=args.scale, t=args.threshold) == 0
points = [list() for _ in range(7)]
def append(x, point):
    points[2*x].append(point[0] / args.scale)
    points[2*x+1].append(height - point[1] / args.scale)
def copy(x, y, first=False):
    points[2*y].append(points[2*x][0 if first else -1])
    points[2*y+1].append(points[2*x+1][0 if first else -1])
def opaque(val):
    points[-1].append(val)
for curve in potrace.Bitmap(bitmap).trace(**{x:vars(args)[x] for x in ('turdsize', 'alphamax', 'opttolerance')}):
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
with open(args.template, 'r') as template:
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
    
with open(args.output, 'w') as output:
    output.write(f"Calc.setState({json.dumps(state, separators=(',', ':'))})")
    print("Successfully created", args.output)
