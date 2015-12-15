import numpy as np

def sphericalToCartesian(spherical):
    r = spherical[0]
    sinTheta = np.sin(spherical[1])
    cosTheta = np.cos(spherical[1])
    sinPhi = np.sin(spherical[2])
    cosPhi = np.cos(spherical[2])

    cartesian = np.zeros(3, np.float32)
    cartesian[0] = r * cosTheta * cosPhi
    cartesian[1] = r * sinTheta * cosPhi
    cartesian[2] = r * sinPhi
    return cartesian;

def cartesianToSpherical(cartesian):
    x, y, z = cartesian

    spherical = np.zeros(3, np.float32)
    spherical[0] = np.sqrt(x*x + y*y + z*z)
    spherical[1] = np.arctan2(y, x)
    if spherical[0] == 0:
        spherical[2] = 0
    else:
        spherical[2] = np.arcsin(z / spherical[0])

    return spherical;

class Transform2D(np.ndarray):
    @classmethod
    def fromVec(cls, vec):
        # Construct a standard numpy array:
        arrayvec = np.array(vec)

        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans

    @classmethod
    def fromParts(cls, vec, angle):
        # Construct a standard numpy array:
        arrayvec = np.array([vec[0], vec[1], angle])

        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans

    @property
    def x(self):
        assert(self.shape[0] == 3)
        return self[0]
    @x.setter
    def x(self, value):
        self[0] = value

    @property
    def y(self):
        assert(self.shape[0] == 3)
        return self[1]
    @y.setter
    def y(self, value):
        self[1] = value

    @property
    def xy(self):
        assert(self.shape[0] == 3)
        return self[0:2]
    @xy.setter
    def xy(self, value):
        assert(self.shape[0] == 3)
        assert(value.shape[0] == 2)
        self[0:2] = value

    @property
    def angle(self):
        assert(self.shape[0] == 3)
        return self[2]
    @angle.setter
    def angle(self, value):
        self[2] = value

    def rotation(self, other):
        assert(self.shape[0] == 3)

class RotatedRectangle:
    def __init__(self, trans2d, size):
        self.trans = Transform2D.fromVec(trans2d)
        self.size = np.array(size)
