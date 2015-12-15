import numpy as np

def rotationFromVectors(dir, up, camera=False):
    dir = dir/np.linalg.norm(dir)
    up = up/np.linalg.norm(up)

    rightRaw = np.cross(dir, up)
    rightLen = np.linalg.norm(rightRaw)
    right = rightRaw/rightLen
    if rightLen < 1e-5:
        print 'ERROR: rotationFromVectors:'
        print '  dir:', dir
        print '  up:', up
        print '  right:', rightRaw
        raise ValueError('Calculated right vector has zero norm.')

    # Recalculate up vector to be orthogonal to the dir vector:
    up = -np.cross(dir, right)
    up = up/np.linalg.norm(up)

    A = None
    if camera:
        # OpenGL eye coordinates:
        #   forward: -Z axis
        #   up: Y axis
        #   right: X-axis
        A = np.column_stack((right, up, -dir))
    else:
        # 'Standard' right handed coordinate system:
        #   forward: X axis
        #   up: Z-axis
        #   right: Y-axis
        A = np.column_stack((dir, right, up))

    # return np.eye(3)
    return np.linalg.inv(A)

def lookAtTransform(pos, target, up, square=False, camera=False):
    pos = np.array(pos, np.float32)
    target = np.array(target, np.float32)
    up = np.array(up, np.float32)

    # print 'lookAtTransform:'
    dir = target - pos

    R = rotationFromVectors(dir, up, camera=camera)

    # print 'R:', R

    pos = np.matrix(pos).T

    V = np.column_stack((R, -R*pos))

    if square:
        V = np.row_stack((
            V,
            np.array([0,0,0,1], np.float32)
        ))

    return V


def openGLPerspectiveMatrix(fovy, aspect, near, far):
    # print 'fovy:', fovy
    # print 'aspect:', aspect
    # print 'near:', near
    # print 'far:', far

    f = 1.0 / np.tan(fovy/2.0)

    # print 'f:', f

    return np.matrix([
    [f/aspect, 0, 0, 0],
    [0, f, 0, 0],
    [0, 0, (far+near)/(near-far), (2.0*near*far)/(near-far)],
    [0, 0, -1, 0]
    ], np.float32)

def viewPortMatrix(framebufferSize):
    xo, yo = (0, 0)
    w, h = framebufferSize

    vpMat = np.matrix([
    [w/2.0, 0, 0, w/2.0 + xo],
    [0, h/2.0, 0, h/2.0 + yo],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
    ], np.float32)

    return vpMat

def convertToOpenGLCameraMatrix(K, framebufferSize, near, far):
    """ Convert a camera calibration matrix into OpenGL format. """
    width, height = framebufferSize

    # print 'framebufferSize:', framebufferSize

    fx = K[0,0]
    fy = K[1,1]
    fovy = 2*np.arctan(0.5*height/fy)#*180/np.pi
    aspect = (width*fy)/(height*fx)
    # define the near and far clipping planes
    # near = 0.1
    # far = 100.0

    # fx = 10.0
    # fy = 10.0
    # fovy = 90*(np.pi/180.0)
    # aspect = (width*fy)/(height*fx)

    proj = openGLPerspectiveMatrix(fovy,aspect,near,far)

    return proj

def buildProjectionMatrix(f, framebufferSize, pos, dir, up):
    w, h = framebufferSize
    cx = w / 2.0
    cy = h / 2.0

    K = np.matrix([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    V = lookAtTransform(pos, pos + dir, up, camera=True)
    # print 'V', V

    return K*V

from pvccamera import Camera as PCVCamera
class Camera(PCVCamera):

    # def __init__(self, P, f, framebufferSize, pos, dir, up, near, far):
    def __init__(self, P, near, far, framebufferSize):
        super(Camera, self).__init__(P)

        # self.f = f
        self.framebufferSize = framebufferSize
        # self.pos = pos
        # self.dir = dir
        # self.up = up
        self.near = near
        self.far = far

    def getOpenGlCameraMatrix(self):
        K, R, t = self.factor()
        # print 'R', R

        # print 'getOpenGlCameraMatrix:'
        # print 'Kraw:', K

        K = convertToOpenGLCameraMatrix(K, self.framebufferSize, self.near, self.far)

        # print 'K:', K

        V = np.column_stack((R, t))
        V = np.row_stack((
            V,
            np.array([0,0,0,1], np.float32)
        ))

        # print 'V:', V

        P = K*V

        # print 'P:', P

        # vpMat = viewPortMatrix(framebufferSize)
        # print 'vpMat:', vpMat
        # print 'VpP:', vpMat*P

        return P

    def updateMatrix(self, f, framebufferSize, pos, dir, up):
        self.P = buildProjectionMatrix(f, framebufferSize, pos, dir, up)

    @classmethod
    def fromMeasurements(cls, f, framebufferSize, pos, dir, up, near, far):

        P = buildProjectionMatrix(f, framebufferSize, pos, dir, up)

        return cls(P, near, far, framebufferSize)

    def unprojectOpenGL(self, u):
        # K, R, t = camera.factor()

        # squareProj = np.row_stack((
        #     camera.P,
        #     np.array([0,0,0,1], np.float32)
        # ))
        # invProj = np.linalg.inv(squareProj)
        # x = invProj*np.row_stack([np.mat(u).T, [1]])
        # x = x[:3]

        # u = np.mat(u).T
        # x = np.linalg.inv(R)*(np.linalg.inv(K)*u - t)

        proj = self.getOpenGlCameraMatrix()
        invProj = np.linalg.inv(proj)
        x = invProj*np.row_stack([np.mat(u).T, [1]])
        x = x[:3] / x[3]
        return x

    def unproject(self, u):
        # K, R, t = camera.factor()

        # squareProj = np.row_stack((
        #     camera.P,
        #     np.array([0,0,0,1], np.float32)
        # ))
        # invProj = np.linalg.inv(squareProj)
        # x = invProj*np.row_stack([np.mat(u).T, [1]])
        # x = x[:3]

        # u = np.mat(u).T
        # x = np.linalg.inv(R)*(np.linalg.inv(K)*u - t)

        proj = self.getOpenGlCameraMatrix()
        invProj = np.linalg.inv(proj)
        x = invProj*np.row_stack([np.mat(u).T, [1]])
        x = x[:3] / x[3]
        return x

    # TODO: Fix handling of camera centre.
    def center(self):
        self.factor()
        self.c = -np.dot(self.R.T,self.t)
        return self.c

    def projectPointToGround(self, xy):
        u = self.unproject([xy[0], xy[1], 0.9])

        self.factor()
        c = self.center()
        d = u - c

        dx, dy, dz = d

        # gu = np.array([u[0], u[1], 0], np.float32)
        gc = np.array([c[0], c[1], 0], np.float32)
        gd = np.array([dx, dy, 0], np.float32)

        if dz > 0:
            return gc + gd*1000

        z = float(c[2])

        t = -z / float(dz)
        gp = gc + gd*t
        return gp
