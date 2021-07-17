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
values = [list() for _ in range(7)]
points = []
def collinear(a, b, c):
    return (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1]) < 1e-3
for curve in potrace.Bitmap(bitmap).trace(**{x:vars(args)[x] for x in ('turdsize', 'alphamax', 'opttolerance')}):
    prev = (0, 2*height) if not points else segment.end_point
    points.append([curve.start_point, curve.start_point, prev, False])
    for segment in curve:
        if segment.is_corner:
            # Attempt to coalesce adjacent line segments
            if points[-1][0] == points[-1][1] and collinear(segment.c, points[-1][0], points[-1][2]) and points[-1][-1]:
                points[-1][:2] = [segment.c, segment.c]
            else:
                points.append([segment.c, segment.c, points[-1][0], True])
            points.append([segment.end_point, segment.end_point, segment.c, True])
        else:
            points.append([segment.end_point, segment.c2, segment.c1, True])

for row in reversed(points):
    for i in range(6):
        values[i].append(row[i//2][i%2] / args.scale)
        if i%2: values[i][-1] = height - values[i][-1]
    values[-1].append(row[-1])


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
for column, nums in zip(table['columns'], values):
    column['values'] = [num2str(num) for num in nums]
    
with open(args.output, 'w') as output:
    output.write(f"Calc.setState({json.dumps(state, separators=(',', ':'))})")
    print("Successfully created", args.output)
