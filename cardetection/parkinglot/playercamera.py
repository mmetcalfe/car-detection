import cyglfw3 as glfw
import numpy as np

from cardetection.carutils.geometry import sphericalToCartesian, cartesianToSpherical

from camera import Camera

class PlayerCamera(Camera):

    def __init__(self, f, framebufferSize, pos, dir, up, near, far):
        cam = Camera.fromMeasurements(f, framebufferSize, pos, dir, up, near, far)
        super(PlayerCamera, self).__init__(cam.P, near, far, framebufferSize)

        self.f = f
        self.pos = np.array(pos, np.float32)
        self.up = np.array(up, np.float32)
        self.sphericalDir = cartesianToSpherical(dir)
        self.sphericalDir[0] = 1
        self.speed = 10.0
        self.lookSpeed = 0.005
        self.lastUpdateTime = 0.0

    def processMoveInput(self, window, deltaTime):
        move = np.zeros(3, np.float32)
        if glfw.GetKey(window, glfw.KEY_W) == glfw.PRESS:
            move[0] += 1
        if glfw.GetKey(window, glfw.KEY_S) == glfw.PRESS:
            move[0] -= 1
        if glfw.GetKey(window, glfw.KEY_D) == glfw.PRESS:
            move[1] += 1
        if glfw.GetKey(window, glfw.KEY_A) == glfw.PRESS:
            move[1] -= 1
        if glfw.GetKey(window, glfw.KEY_E) == glfw.PRESS:
            move[2] += 1
        if glfw.GetKey(window, glfw.KEY_Q) == glfw.PRESS:
            move[2] -= 1

        if np.linalg.norm(move) > 0:
            move = move / np.linalg.norm(move)

        if glfw.GetKey(window, glfw.KEY_LEFT_SHIFT):
            move *= 5

        if glfw.GetKey(window, glfw.KEY_LEFT_CONTROL):
            move *= 0.2

        dir = sphericalToCartesian(self.sphericalDir)
        right = np.cross(dir, self.up)

        forwardVec = dir * move[0] * self.speed * deltaTime
        rightVec = right * move[1] * self.speed * deltaTime
        upVec = self.up * move[2] * self.speed * deltaTime

        self.pos += forwardVec + rightVec + upVec

    def processLookInput(self, window, deltaTime):

        cursor = np.array(glfw.GetCursorPos(window), np.float32)
        if 'lastCursor' in dir(self):
            deltaMouse = cursor - self.lastCursor
        else:
            deltaMouse = np.zeros(2)
        self.lastCursor = cursor

        # windowWidth, windowHeight = glfw.GetWindowSize(window)
        # glfw.SetCursorPos(window, windowWidth / 2, windowHeight / 2)

        self.sphericalDir[1] -= float(deltaMouse[0]) * self.lookSpeed
        self.sphericalDir[2] -= float(deltaMouse[1]) * self.lookSpeed
        # self.sphericalDir[1:3] -= deltaMouse * self.lookSpeed
        self.sphericalDir[2] = min(max(-np.pi*0.5 * 0.9, self.sphericalDir[2]), np.pi*0.5 * 0.9)

    def processPlayerInput(self, window):
        currentTime = glfw.GetTime()
        deltaTime = float(currentTime - self.lastUpdateTime)
        self.lastUpdateTime = currentTime

        # print 'self.pos', self.pos
        # print 'self.sphericalDir', self.sphericalDir

        self.processMoveInput(window, deltaTime)
        self.processLookInput(window, deltaTime)

        dir = sphericalToCartesian(self.sphericalDir)
        self.updateMatrix(self.f, self.framebufferSize, self.pos, dir, self.up)
