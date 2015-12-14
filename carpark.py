import math
import cairo
import numpy as np
from geometry import *
from drawing2d import ExtendedCairoContext

class Vehicle:
    def __init__(self, position, rotation, boxSize):
        assert(boxSize.shape[0] == 3)
        assert(position.shape[0] == 3)
        assert(rotation.shape[0] == 3)

        self.boxSize = np.array(boxSize)
        self.position = np.array(position)
        self.rotation = np.array(rotation)

class ParkingSpace(RotatedRectangle):
    pass

class ParkingLot:
    def __init__(self, size, centre, canvasSize):
        assert(size.shape[0] == 2)
        assert(centre.shape[0] == 2)
        assert(canvasSize.shape[0] == 2)
        self.size = np.array(size)
        self.centre = np.array(centre)
        self.canvasSize = np.array(canvasSize)
        self.spaces = []


# Carpark config:
mapSize = np.array([5.0, 3.0])
mapCentre = np.array([0.5, 0.75])

# Drawing config:
canvasSize = np.array([500, 300])

#####

# Create a canvas of the specified size:
surface = cairo.PDFSurface ("carpark.pdf", canvasSize[0], canvasSize[1])
# ctx = cairo.Context(surface)
ctx = ExtendedCairoContext(surface)

ctx.transformToRealWorldUnits(canvasSize, mapSize, mapCentre)

ctx.set_line_width(0.05)
ctx.setCol(ctx.getRandCol())

# Draw map outline:
trans = Transform2D.fromParts(mapSize/2-mapCentre, 0)
rect = RotatedRectangle(trans, mapSize)
ctx.rotatedRectangle(rect)
ctx.stroke()

# setCol(ctx, randCol())
# ctx.rectangle(-0.1,-0.1,0.2,0.2)
# ctx.fill()
#
# setCol(ctx, randCol())
# ctx.rectangle(0.5,0.25,0.5,0.5)
# ctx.fill()

# ctx.set_source_rgb(1,0,0)
# ctx.move_to(width/2,height/2)
# ctx.arc(width/2,height/2,512*0.25,0,math.pi*2)
# ctx.fill()

ctx.show_page()

# ctx.destroy()
# surface.destroy()
