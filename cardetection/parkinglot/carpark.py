import math
import numpy as np
from geometry import *

class Vehicle(object):
    def __init__(self, position, rotation, boxSize):
        assert(boxSize.shape[0] == 3)
        assert(position.shape[0] == 3)
        assert(rotation.shape[0] == 3)

        self.boxSize = np.array(boxSize)
        self.position = np.array(position)
        self.rotation = np.array(rotation)

class ParkingSpace(RotatedRectangle):
    """
    A rectangle in which a single car may park.
    """
    pass

# class ParkingArea(RotatedRectangle):
#     """
#     An area in which any number of cars may park
#     """
#     pass

class ParkingLot(object):
    def __init__(self, size, centre, canvasSize):
        assert(size.shape[0] == 2)
        assert(centre.shape[0] == 2)
        assert(canvasSize.shape[0] == 2)
        self.size = np.array(size)
        self.centre = np.array(centre)
        self.canvasSize = np.array(canvasSize)
        self.spaces = []
        self.cameras = []
        self.detections = []

class Detection(RotatedRectangle):
    """
    The axis-aligned rectangular bounding box of a detected object.
    """
    pass
