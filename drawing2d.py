import numpy as np
import cairo
from geometry import *

def getRandCol():
    vc = np.random.rand(3)
    ln = np.linalg.norm(vc)
    return vc / ln

class ExtendedCairoContext(cairo.Context):
    def numpy2CairoMat(numpyMat):
        assert(numpyMat[2,0] == 0)
        assert(numpyMat[2,1] == 0)

        return cairo.Matrix(
              xx = numpyMat[0,0]
            , yx = numpyMat[1,0]
            , xy = numpyMat[0,1]
            , yy = numpyMat[1,1]
            , x0 = numpyMat[0,2]
            , y0 = numpyMat[1,2]
            )

    def transformToLocal(self, trans2d):
        self.translate(trans2d.x, trans2d.y)
        self.rotate(trans2d.angle)

    def transformToRealWorldUnits(self, canvasSize, realSize, realCentre):
        # Transform the canvas to allow drawing in real-world units:
        canvasAspect = canvasSize[0] / canvasSize[1];
        mapAspect = realSize[0] / realSize[1];
        if mapAspect > canvasAspect:
            drawingScale = canvasSize[0] / realSize[0];
            self.scale(drawingScale, drawingScale);
        else:
            drawingScale = canvasSize[1] / realSize[1];
            self.scale(drawingScale, drawingScale);

        # Set the coordinate system origin as specified:
        # ctx.translate(realCentre[0], 0)
        self.translate(realCentre[0], realSize[1] - realCentre[1])

        # Make the positive y-axis point upwards:
        self.scale(1.0,-1.0)

    def getRandCol(self):
        return getRandCol()

    def setCol(self, colVec):
        self.set_source_rgb(colVec[0], colVec[1], colVec[2])

    def rotatedRectangle(self, rotRect):
        self.save()
        self.transformToLocal(rotRect.trans)
        self.scale(rotRect.size[0], rotRect.size[1])
        self.rectangle(-0.5,-0.5,1,1)
        self.restore()

    def circle(self, centre, radius):
        ctx.arc(centre[0], centre[1], radius, 0, math.pi*2)
