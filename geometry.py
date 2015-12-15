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

def normaliseAngle(value):
    angle = np.fmod(value, 2 * np.pi);

    if (angle <= -np.pi):
        angle += np.pi * 2;

    if (angle > np.pi):
        angle -= 2 * np.pi;

    return angle;

class Transform2D(np.ndarray):
    @classmethod
    def fromVec(cls, vec):
        assert(len(vec) == 3)
        # Construct a standard numpy array:
        arrayvec = np.array(vec)
        # arrayvec = np.array([vec[0], vec[1], vec[2]])

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

    def worldToLocal(self, reference):
        reference = Transform2D.fromVec(reference)
        cosAngle = np.cos(self.angle)
        sinAngle = np.sin(self.angle)
        diff = reference - self
        # translates to rotZ(this.angle) * (reference - this)
        return Transform2D.fromVec([
            cosAngle * diff.x + sinAngle * diff.y,
            -sinAngle * diff.x + cosAngle * diff.y,
            normaliseAngle(diff.angle)
        ])

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

    def contains(self, pt):
        local = self.trans.worldToLocal([pt[0], pt[1], 0]);
        absLocal = np.abs(local.xy);

        if absLocal[0] > self.size[0]*0.5 or absLocal[1] > self.size[1]*0.5:
            return False;

        return True

def intersectRectTri(rect, tri, ctx):
    for u in np.arange(0, 1, 0.05):
        for v in np.arange(0, 1, 0.05):
            if u + v > 1:
                continue
            w = 1.0 - u - v
            q = u * tri[0] + v * tri[1] + w * tri[2]
            # ctx.circle(q, 0.02)
            # ctx.fill()
            if rect.contains(q):
                return True
    return False

def intersectRectangleConvexQuad(rect, pts, ctx):
    if intersectRectTri(rect, [pts[0], pts[1], pts[2]], ctx):
        return True
    if intersectRectTri(rect, [pts[2], pts[3], pts[0]], ctx):
        return True

    return False
