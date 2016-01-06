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
        # Note: Also allow different canvas and map aspect ratios.
        canvasAspect = canvasSize[0] / float(canvasSize[1])
        mapAspect = realSize[0] / float(realSize[1])
        if mapAspect > canvasAspect:
            drawingScale = canvasSize[0] / float(realSize[0])
            self.scale(drawingScale, drawingScale);
        else:
            drawingScale = canvasSize[1] / float(realSize[1])
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
        self.arc(centre[0], centre[1], radius, 0, np.pi*2)

    def moveTo(self, pt):
        self.move_to(pt[0], pt[1])
    def lineTo(self, pt):
        self.line_to(pt[0], pt[1])
    def line(self, frm, to):
        self.moveTo(cr, frm);
        self.lineTo(cr, to);

    # ExtendedCairoContext.graph :: RotatedRectangle -> [Transform2D] -> Float -> IO ()
    def graph(self, boundingBox, values, yRange):
        self.save(cr)
        self.set_line_width(cr, yRange*0.002)
        self.setCol(cr, [0.5, 0.5, 0.5])
        self.rotatedRectangle(cr, boundingBox)
        self.stroke(cr)

        self.transformToLocal(cr, boundingBox.trans)
        xw = boundingBox.size[0]
        yw = boundingBox.size[1]
        axisSpace = 0.1 * xw
        xZero = -xw/2 + axisSpace
        xRange = xw - axisSpace

        # double minValue = arma::datum::inf
        # double maxValue = -arma::datum::inf
        # for (auto v : values) {
        #     minValue = std::min(minValue, arma::min(v))
        #     maxValue = std::max(maxValue, arma::max(v))
        # }
        # double yRange = maxValue - minValue

        self.transformToLocal(cr, [xZero, 0, 0])
        self.scale(cr, 1, -1)

        self.setCol(cr, [0.0, 0.0, 0.0])
        self.line(cr, [0,0], [xRange,0])
        self.line(cr, [0,-yw/2], [0,yw/2])
        self.stroke(cr)

        yScale = (yw/2)/yRange
        x = 0
        xStep = xRange / len(values)
        for v in values:
            x += xStep
            for e in v:
                self.setCol(cr, [1.0, 0.0, 0.0])
                self.circle(cr, [x, e*yScale], yw*0.02)
                self.fill(cr)

        self.restore(cr)
