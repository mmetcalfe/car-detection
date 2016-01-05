from sklearn import svm
import numpy as np

data_vectors = []
data_labels = []

for i in xrange(0, 5000):
    vc = np.random.rand(2)*2 - 1
    vx, vy = vc[:2]
    y = np.sin(5*vx)*0.6
    # y = np.sin(vx)

    dist = vy - y

    vl = 0 if dist < 0 else 1

    # # Fuzzy border:
    # if np.abs(dist) < 0.2:
        # continue
    #     if dist > 0:
    #         vl = 1 - vl
    #     vl = np.round(np.random.rand(1)[0])
    #     # continue

    # Random flipping:
    # if np.random.rand(1) < 0.1:
    if np.random.rand(1)*np.abs(dist) < 0.05:
        vl = np.round(np.random.rand(1)[0])
        # vl = 1 - vl


    # # Undersample one class:
    # if vl == 0:
    #     if np.random.rand(1) < 0.9:
    #         continue


    data_vectors.append(vc)
    data_labels.append(vl)

data_vectors = np.array(data_vectors)
data_labels = np.array(data_labels)

# print 'data_vectors:', len(data_vectors), data_vectors
# print 'data_labels:', len(data_labels), data_labels

# clf = svm.SVC()
# clf = svm.SVC(C=10, kernel='sigmoid', gamma=5, coef0=-7)
# clf = svm.LinearSVC(C=1)
# clf = svm.SVC(C=1, kernel='rbf', gamma=2, class_weight={0:1, 1:0.25})
clf = svm.SVC(C=1, kernel='rbf', gamma=1)
# clf = svm.SVC(C=1, kernel='linear')
# clf = svm.SVC(C=1, kernel='poly', degree=3, coef0=0.1)
clf.fit(data_vectors, data_labels)

# print 'clf.support_vectors_', clf.support_vectors_
# print 'clf.support_', clf.support_
# print 'clf.n_support_', clf.n_support_

# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

pts = 100
query_points = []
for x in np.linspace(-1, 1, pts):
    for y in np.linspace(-1, 1, pts):
        query_points.append([x,y])
predictions = clf.predict(query_points)

import cairo
from drawing2d import ExtendedCairoContext
from geometry import *

# Carpark config:
mapSize = np.array([2.0, 2.0])
mapCentre = np.array([1.0, 1.0])

# Drawing config:
canvasSize = np.array([500, 500])

# Create a canvas of the specified size:
surface = cairo.PDFSurface('svmtest.pdf', canvasSize[0], canvasSize[1])
# ctx = cairo.Context(surface)
ctx = ExtendedCairoContext(surface)
ctx.transformToRealWorldUnits(canvasSize, mapSize, mapCentre)

lw = 0.05 * 0.1

# Draw map outline:
ctx.set_line_width(lw)
ctx.setCol(ctx.getRandCol())
trans = Transform2D.fromParts(mapSize/2-mapCentre, 0)
rect = RotatedRectangle(trans, mapSize)
ctx.rotatedRectangle(rect)
ctx.stroke()


posCol = ctx.getRandCol()
negCol = 1 - posCol # ctx.getRandCol()

# Draw predictions:
ctx.set_line_width(lw)
for qv, ql in zip(query_points, predictions):
    ctx.setCol(posCol if ql > 0 else negCol)
    rs = 2.0 / pts
    ctx.rotatedRectangle(RotatedRectangle([qv[0], qv[1], 0], [rs, rs]))
    # ctx.circle([qv[0], qv[1]], 0.01)
    ctx.fill()

# Draw support vectors:
if 'support_' in dir(clf):
    ctx.set_line_width(lw*0.2)
    for svi in clf.support_:
        dv = data_vectors[svi]
        dl = data_labels[svi]
        ptCol = posCol if dl < 0 else negCol
        ctx.setCol(1 - ptCol*0.1)
        ctx.circle([dv[0], dv[1]], lw*3.75)
        ctx.fill()

# Draw training samples:
ctx.set_line_width(0.01)
for dv, dl in zip(data_vectors, data_labels):
    ptCol = posCol if dl > 0 else negCol
    ctx.setCol(ptCol)
    ctx.circle([dv[0], dv[1]], lw*2)
    ctx.fill_preserve()
    ctx.setCol(ptCol*0.2)
    ctx.stroke()

ctx.show_page()
