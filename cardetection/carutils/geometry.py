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

class RotatedRectangle(object):
    def __init__(self, trans2d, size):
        self.trans = Transform2D.fromVec(trans2d)
        self.size = np.array(size)

    def contains(self, pt):
        local = self.trans.worldToLocal([pt[0], pt[1], 0]);
        absLocal = np.abs(local.xy);

        if absLocal[0] > self.size[0]*0.5 or absLocal[1] > self.size[1]*0.5:
            return False;

        return True

# intersectRectangleConvexQuad :: RotatedRectangle -> [np.array] -> Bool
def intersectRectTri(rrect, tri, ctx):
    # Just sample a bunch of points.
    # Note: This could obviously be made much faster/more accurate.
    for u in np.arange(0, 1, 0.05):
        for v in np.arange(0, 1, 0.05):
            if u + v > 1:
                continue
            w = 1.0 - u - v
            q = u * tri[0] + v * tri[1] + w * tri[2]
            # ctx.circle(q, 0.02)
            # ctx.fill()
            if rrect.contains(q):
                return True
    return False

# intersectRectangleConvexQuad :: RotatedRectangle -> [np.array] -> Bool
def intersectRectangleConvexQuad(rrect, pts, ctx):
    if intersectRectTri(rrect, [pts[0], pts[1], pts[2]], ctx):
        return True
    if intersectRectTri(rrect, [pts[2], pts[3], pts[0]], ctx):
        return True

    return False

class Rectangle(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return '{{Rectangle | x:{}, y:{}, w:{}, h:{}}}'.format(self.x, self.y, self.w, self.h)

    # Rectangle.fromList :: [Int] -> Rectangle
    @classmethod
    def fromList(cls, xywh):
        x, y, w, h = xywh
        return cls(x, y, w, h)

    # # Rectangle.fromPixelRanges :: Float -> Float -> Float -> Float -> Rectangle
    # @classmethod
    # def fromPixelRanges(cls, x1, y1, x2, y2):
    #     w = x2 - x1 + 1
    #     h = y2 - y1 + 1
    #     return cls(x1, y1, w, h)

class PixelRetangle(np.ndarray):
    @classmethod
    def fromCoords(cls, x1, y1, x2, y2):
        # Construct a standard numpy array:
        arrayvec = np.array([x1, y1, x2, y2])
        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans
    @classmethod
    def fromCorners(cls, tl, br):
        assert(len(tl) == 2)
        assert(len(br) == 2)
        # Construct a standard numpy array:
        arrayvec = np.array([tl[0], tl[1], br[0], br[1]])
        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans
    @property
    def x1(self):
        assert(self.shape[0] == 4)
        return self[0]
    @x1.setter
    def x1(self, value):
        assert(self.shape[0] == 4)
        self[0] = value
    @property
    def y1(self):
        assert(self.shape[0] == 4)
        return self[1]
    @y1.setter
    def y1(self, value):
        assert(self.shape[0] == 4)
        self[1] = value
    @property
    def x2(self):
        assert(self.shape[0] == 4)
        return self[2]
    @x2.setter
    def x2(self, value):
        assert(self.shape[0] == 4)
        self[2] = value
    @property
    def y2(self):
        assert(self.shape[0] == 4)
        return self[3]
    @y2.setter
    def y2(self, value):
        assert(self.shape[0] == 4)
        self[3] = value

    @property
    def tl(self):
        assert(self.shape[0] == 4)
        return self[0:2]
    @tl.setter
    def tl(self, value):
        assert(self.shape[0] == 4)
        assert(len(value) == 2)
        self[0:2] = value
    @property
    def br(self):
        assert(self.shape[0] == 4)
        return self[2:4]
    @br.setter
    def br(self, value):
        assert(self.shape[0] == 4)
        assert(len(value) == 2)
        self[2:4] = value

    # If this rectangle lay within a frame of the given shape, and that frame
    # were to be rotated by 180 degrees along with the rectangle, return the
    # rectangle that would result.
    def flipped(self, dimensions):
        dims = np.array(dimensions)
        tl = dims - self.tl
        br = dims - self.br
        return PixelRetangle.fromCorners(tl, br)

# extendBoundingBox :: Rectangle -> Float -> Rectangle
def extendBoundingBox(rect, new_aspect):
    new_aspect = float(new_aspect)
    aspect = rect.w / float(rect.h)

    # New dimensions from aspect ratio:
    w, h = rect.w, rect.h
    if new_aspect >= aspect:
        w = int(np.round(rect.h * new_aspect))
    else:
        h = int(np.round(rect.w / new_aspect))

    # Old centre:
    cx = rect.x + (rect.w - 1) / 2.0
    cy = rect.y + (rect.h - 1) / 2.0
    # New corner from centre and new width:
    x = int(np.round(cx - (w - 1) / 2.0))
    y = int(np.round(cy - (h - 1) / 2.0))
    return Rectangle(x, y, w, h)

# padBoundingBox :: Rectangle -> (Float, Float) -> Rectangle
def padBoundingBox(rect, padding_fracs):
    w = int(np.round(rect.w * (1.0 + padding_fracs[0])))
    h = int(np.round(rect.h * (1.0 + padding_fracs[1])))

    # Old centre:
    cx = rect.x + (rect.w - 1) / 2.0
    cy = rect.y + (rect.h - 1) / 2.0
    # New corner from centre and new width:
    x = int(np.round(cx - (w - 1) / 2.0))
    y = int(np.round(cy - (h - 1) / 2.0))
    return Rectangle(x, y, w, h)
