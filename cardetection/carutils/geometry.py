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

class PixelRectangle(np.ndarray):
    @classmethod
    def fromCoords(cls, x1, y1, x2, y2):
        # Construct a standard numpy array:
        arrayvec = np.array([x1, y1, x2, y2])
        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans
    @classmethod
    def fromCorners(cls, c1, c2):
        assert(len(c1) == 2)
        assert(len(c2) == 2)

        # Find the top-left and bottom-right corner:
        tl = np.minimum(c1, c2)
        br = np.maximum(c1, c2)

        # Construct a standard numpy array:
        arrayvec = np.array([tl[0], tl[1], br[0], br[1]])
        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        return trans

    @classmethod
    def from_opencv_bbox(cls, bbox):
        assert(len(bbox) == 4)
        # Construct a standard numpy array:
        arrayvec = np.array([0, 0, 0, 0])
        # Perform a numpy 'view cast' to the target type 'cls':
        trans = arrayvec.view(cls)
        trans.opencv_bbox = bbox
        return trans

    @classmethod
    def random(cls, img_dims):
        assert(len(img_dims) == 2)
        w, h = img_dims
        x1, x2 = np.random.randint(0, w, 2)
        y1, y2 = np.random.randint(0, h, 2)
        return cls.fromCorners([x1, y1], [x2, y2])

    @classmethod
    def random_with_same_aspect(cls, window_dims, img_dims, min_side_length=48):
        assert(len(img_dims) == 2 and len(window_dims) == 2)
        assert(window_dims[0] <= img_dims[0] and window_dims[1] <= img_dims[1])

        window_dims = np.array(window_dims, dtype=np.float)
        aspect = window_dims[1] / float(window_dims[0])

        ww, wh = window_dims
        sf = min_side_length / min(ww, wh)
        raw_min_size = map(int, np.round(window_dims*sf))
        min_size = np.maximum(raw_min_size, [min_side_length, min_side_length])

        w, h = img_dims
        min_w, min_h = min_size

        aspect = min_w / float(min_h)

        max_w = min(w, int(h*aspect))
        max_h = min(h, int(max_w/aspect))
        assert(abs(aspect - max_w/float(max_h)) < 0.1)

        # Prefer smaller rectangles:
        # fn = lambda x: 1.0/x
        # fn_inv = lambda x: 1.0/x
        # fn_reg_w = np.random.uniform(fn(min_w), fn(max_w), 1)
        # reg_w = max(1, int(fn_inv(fn_reg_w)))

        # Prefer smaller rectangles:
        # Note: We sample from a distribution that has been chosen and tuned to
        #       give qualitatively acceptable results.
        #       i.e. It chooses larger rectangles less often than smaller
        #       rectangles (because they tend to overlap more and so are more similar).
        # See trainhog.test_random_with_same_aspect for a visualisation.
        a = 0.5
        b = 2
        r = max_w - min_w
        reg_w_frac = np.random.beta(a, b, 1)
        reg_w = max(1, int(min_w + r*reg_w_frac))

        # Uniform sampling:
        # reg_w = np.random.random_integers(min_w, max_w, 1)

        reg_h = int(reg_w / aspect)
        assert(abs(aspect - reg_w/float(reg_h)) < 0.2)

        if w - reg_w <= 0:
            print img_dims, (reg_w, reg_h)
        x1 = np.random.random_integers(0, w - reg_w, 1)
        y1 = np.random.random_integers(0, h - reg_h, 1)
        x2 = x1 + reg_w - 1
        y2 = y1 + reg_h - 1
        return cls.fromCorners([x1, y1], [x2, y2])

    @property
    def x1(self):
        assert(self.shape[0] == 4)
        return int(self[0])
    @x1.setter
    def x1(self, value):
        assert(self.shape[0] == 4)
        self[0] = value
    @property
    def y1(self):
        assert(self.shape[0] == 4)
        return int(self[1])
    @y1.setter
    def y1(self, value):
        assert(self.shape[0] == 4)
        self[1] = value
    @property
    def x2(self):
        assert(self.shape[0] == 4)
        return int(self[2])
    @x2.setter
    def x2(self, value):
        assert(self.shape[0] == 4)
        self[2] = value
    @property
    def y2(self):
        assert(self.shape[0] == 4)
        return int(self[3])
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

    @property
    def w(self):
        assert(self.shape[0] == 4)
        return int(self.x2 - self.x1 + 1)
    @property
    def h(self):
        assert(self.shape[0] == 4)
        return int(self.y2 - self.y1 + 1)
    @property
    def aspect(self):
        assert(self.shape[0] == 4)
        return self.w / float(self.h)

    @property
    def exact_centre(self):
        cx = self.x1 + (self.w - 1) / 2.0
        cy = self.y1 + (self.h - 1) / 2.0
        return (cx, cy)

    # If this rectangle lay within a frame of the given shape, and that frame
    # were to be rotated by 180 degrees along with the rectangle, return the
    # rectangle that would result.
    def rotated_180(self, dimensions):
        dims = np.array(dimensions)
        tl = dims - self.tl
        br = dims - self.br
        return PixelRectangle.fromCorners(tl, br)

    # If this rectangle lay within a frame of the given shape, and that frame
    # were to be mirrored about the x-axis along with the rectangle, return the
    # rectangle that would result.
    def mirrored_x(self, dimensions):
        w, h_ = dimensions
        x1 = w - self.x2
        x2 = w - self.x1
        return PixelRectangle.fromCoords(x1, self.y1, x2, self.y2)

    # Translate this rectangle by the given vector while maintaining its width
    # and height, and keeping it within the image frame.
    # i.e. The full translation vector may not be applied if the rectangle is
    # near the edge of the frame.
    def translated(self, trans_vec, dimensions):
        tx, ty = map(int, np.round(trans_vec))
        w, h = dimensions

        # Modify translation to keep rectangle within the frame:
        # Note: Deliberately perform all corrections when the translation vector
        # is (0, 0) so that this method can be used for correcting bad
        # rectangles.
        if tx >= 0:
            if self.x2 + tx >= w:
                tx = (w - 1) - self.x2
        if tx <= 0:
            if self.x1 + tx < 0:
                tx = 0 - self.x1
        if ty >= 0:
            if self.y2 + ty >= h:
                ty = (h - 1) - self.y2
        if ty <= 0:
            if self.y1 + ty < 0:
                ty = 0 - self.y1

        # Apply the translation:
        x1 = self.x1 + tx
        x2 = self.x2 + tx
        y1 = self.y1 + ty
        y2 = self.y2 + ty

        return PixelRectangle.fromCoords(x1, y1, x2, y2)

    # Returns whether this rectangle is completely contained within an image
    # frame of the given dimensions.
    #
    # PixelRectangle.lies_within_frame :: (Int, Int) -> Bool
    def lies_within_frame(self, dimensions):
        if self.x1 < 0 or self.y1 < 0:
            return False
        if self.x2 > dimensions[0] or self.y2 > dimensions[1]:
            return False
        return True

    # Returns whether this rectangle includes pixels on the very edges of an
    # image frame with the given dimensions.
    #
    # PixelRectangle.touches_frame_edge :: (Int, Int) -> Bool
    def touches_frame_edge(self, dimensions):
        if self.x1 == 0 or self.y1 == 0:
            return True
        if self.x2 == dimensions[0] - 1 or self.y2 == dimensions[1] - 1:
            return True
        return False

    # Return this rectangle with its width or height increased such that its
    # aspect ratio matches the given aspect ratio.
    #
    # If the scaling casuses part of the rectangle to leave the image
    # boundaries, the rectangle is translated back into the image boundaries.
    #
    # PixelRectangle.enlarge_to_aspect :: Float -> PixelRectangle
    def enlarge_to_aspect(self, new_aspect):
        new_aspect = float(new_aspect)
        aspect = self.w / float(self.h)

        # New dimensions from aspect ratio:
        w, h = self.w, self.h
        if new_aspect >= aspect:
            w = int(np.round(self.h * new_aspect))
        else:
            h = int(np.round(self.w / new_aspect))

        # Old centre:
        cx, cy = self.exact_centre
        # New corner from centre and new width:
        x = int(np.round(cx - (w - 1) / 2.0))
        y = int(np.round(cy - (h - 1) / 2.0))
        new_rect = PixelRectangle.from_opencv_bbox([x, y, w, h])

        return new_rect

    # Return this rectangle with its width and height scaled by the scale
    # factors, and placed such that its centre is as close as possible to its
    # original location.
    #
    # If the scaling would cause any dimension of the rectangle to grow larger
    # than the image frame, that dimension is clamped to the image dimension,
    # possible changing the rectangle's aspect ratio.
    #
    # If the scaling casuses part of the rectangle to leave the image
    # boundaries, the rectangle is translated back into the image boundaries.
    #
    # PixelRectangle.scaled_about_center :: (Float, Float) -> (Int, Int) -> PixelRectangle
    def scaled_about_center(self, scale_factors, dimensions):
        # Scale the current dimensions:
        scaled_w = int(np.round(self.w * scale_factors[0]))
        scaled_h = int(np.round(self.h * scale_factors[1]))

        # Ensure that new dimensions fit inside the image frame:
        max_w, max_h = dimensions
        w = min(max_w, max(2, scaled_w))
        h = min(max_h, max(2, scaled_h))

        # Old centre:
        cx, cy = self.exact_centre
        # New corner from centre and new width:
        x = int(np.round(cx - (w - 1) / 2.0))
        y = int(np.round(cy - (h - 1) / 2.0))
        new_rect = PixelRectangle.from_opencv_bbox([x, y, w, h])

        # Translate the rectangle such that it is entirely contained within the
        # frame:
        return new_rect.translated([0, 0], dimensions)

    # Return the rectangle that would result if an image containing this
    # rectangle was scaled (along with the rectangle) from img_dims to have the
    # dimensions new_dims.
    def scale_image(self, img_dims, new_dims):
        xs = new_dims[0] / float(img_dims[0])
        ys = new_dims[1] / float(img_dims[1])

        x1 = int(self.x1 * xs)
        x2 = int(self.x2 * xs)
        y1 = int(self.y1 * ys)
        y2 = int(self.y2 * ys)

        tl = np.array([x1, y1])
        br = np.array([x2, y2])
        tl = np.minimum(np.maximum(tl, [0, 0]), new_dims - np.array([1, 1]))
        br = np.minimum(np.maximum(br, [0, 0]), new_dims - np.array([1, 1]))

        return PixelRectangle.fromCorners(tl, br)

    @property
    def opencv_bbox(self):
        # Note: Must add 1 to the difference, since the ranges are inclusive.
        w = self.x2 - self.x1 + 1
        h = self.y2 - self.y1 + 1
        lst = [self.x1, self.y1, w, h]
        return map(int, lst)

    @opencv_bbox.setter
    def opencv_bbox(self, xywh):
        x, y, w, h = xywh
        self.x1 = x
        self.y1 = y
        # Note the subtraction, since (x,y) is the top-left grid cell (pixel),
        # axis-aligned width is specified in full grid cells:
        self.x2 = x + w - 1
        self.y2 = y + h - 1

    def contains(self, pt):
        x, y = pt

        if x < self.x1 or self.x2 < x:
            return False;

        if y < self.y1 or self.y2 < y:
            return False;

        return True

    def intersects_pixelrectangle(self, other):
        if self.x2 < other.x1:
            return False
        if other.x2 < self.x1:
            return False
        if self.y2 < other.y1:
            return False
        if other.y2 < self.y1:
            return False

        return True

    # Return minimum distance between this rectangle and the given rectangle.
    # If the rectangles overlap, returns a negative value indicating the size of
    # the overlap.
    def distance_pixelrectangle(self, other):
        # If x-overlap and not y-overlap:
        x_dist = min(abs(other.x1 - self.x2), abs(self.x1 - other.x2))
        y_dist = min(abs(other.y1 - self.y2), abs(self.y1 - other.y2))

        x_overlap = not (self.x2 < other.x1 and other.x2 < self.x1)
        y_overlap = not (self.y2 < other.y1 and other.y2 < self.y1)

        if x_overlap and not y_overlap:
            return x_dist

        if y_overlap and not x_overlap:
            return y_dist

        if not x_overlap and not y_overlap:
            # Corner-to-corner distance:
            return np.sqrt(x_dist*x_dist + y_dist*y_dist)

        # The rectangles overlap.
        # Return the minimum overlap as a negative integer:
        x_dist = max(other.x1 - self.x2, self.x1 - other.x2)
        y_dist = max(other.y1 - self.y2, self.y1 - other.y2)
        return max(x_dist, y_dist)

    # Return the rectangle that results from translating this rectangle the
    # shortest distance horizontally or vertically such that it does not
    # intersect the given rectangle.
    def moved_to_clear(self, other, return_offset=False):
        if not self.intersects_pixelrectangle(other):
            return PixelRectangle.from_opencv_bbox(self.opencv_bbox)

        left_offset = self.x2 - other.x1 + 1
        right_offset = self.x1 - other.x2 - 1
        up_offset = self.y2 - other.y1 + 1
        down_offset = self.y1 - other.y2 - 1

        x_dist = min(abs(left_offset), abs(right_offset))
        y_dist = min(abs(up_offset), abs(down_offset))

        dx = 0
        dy = 0

        if x_dist < y_dist:
            dx = left_offset if abs(left_offset) < abs(right_offset) else right_offset
        else:
            dy = up_offset if abs(up_offset) < abs(down_offset) else down_offset

        x, y, w, h = self.opencv_bbox
        rect = PixelRectangle.from_opencv_bbox([x-dx, y-dy, w, h])
        if not return_offset:
            return rect
        else:
            offset = (-dx, -dy)
            return rect, offset


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
