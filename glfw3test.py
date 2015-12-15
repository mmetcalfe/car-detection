import os
import sys

from pprint import pprint

# needed if you're running the OS-X system python
try:
    from AppKit import NSApp, NSApplication
except:
    pass

import cyglfw3 as glfw
# https://github.com/adamlwgriffiths/cyglfw3
# import OpenGL.GL as GL
from OpenGL.GL import *
# from math import *

from cbwindow import CbWindow

# from render import renderScene, renderTest
from openglscene import Scene
from carparkrender import ParkingLotRender
from carpark import *
from playercamera import PlayerCamera

def processCameraSelectInput(window, parkingLot):
    if glfw.GetKey(window.window, glfw.KEY_GRAVE_ACCENT) == glfw.PRESS:
        window.currentCamera = window.playerCamera
        return

    camIndex = -1
    for key_id in range(glfw.KEY_9, glfw.KEY_0, -1):
        if glfw.GetKey(window.window, key_id) == glfw.PRESS:
            keyNum = key_id - glfw.KEY_0
            camIndex = keyNum - 1

    if camIndex >= 0 and camIndex < len(parkingLot.cameras):
        window.currentCamera = parkingLot.cameras[camIndex]

def buildParkingLot():
    parkingLot = ParkingLot(
        # size = np.array([5.0, 3.0]),
        # centre = np.array([2.5, 1.5]),
        size = np.array([7.0, 5.0]),
        centre = np.array([3, 3]),
        canvasSize = np.array([700, 500])
    )

    # Add parking spaces:
    parkingLot.spaces.append(ParkingSpace([2, 0.5, 0], [0.8, 1.8]))
    parkingLot.spaces.append(ParkingSpace([1, 0.5, 0], [0.8, 1.8]))
    parkingLot.spaces.append(ParkingSpace([0, 0.5, 0], [0.8, 1.8]))

    # Add cameras:
    f = 240
    up = np.array([0, 0, 1])
    near = 0.2
    far = 100
    framebufferSize = (500, 500)

    pos = np.array([3, -2, 3])
    dir = np.array([-3, 2, -3])
    parkingLot.cameras.append(PlayerCamera(f, framebufferSize, pos, dir, up, near, far))
    pos = np.array([-2, -1, 4])
    dir = np.array([2, 1, -3])
    parkingLot.cameras.append(PlayerCamera(f, framebufferSize, pos, dir, up, near, far))

    return parkingLot

def main():
    cairoSavePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'carpark.pdf')
    # cairoSavePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'carpark.pdf')
    # cairoSavePath = os.path.join(os.path.basename(__file__), 'carpark.pdf')
    # cairoSavePath = os.path.join(os.getcwd(), 'carpark.pdf')

    if not glfw.Init():
        exit()

    window = CbWindow(640, 480, 'window')

    if not window.window:
        glfw.Terminate()
        print('GLFW: Failed to create window')
        exit()

    print glGetString(GL_VERSION)

    parkingLot = buildParkingLot()
    mainRender = ParkingLotRender(parkingLot)
    window.mainRender = mainRender

    mainRender.renderCairo(window.playerCamera, cairoSavePath)

    # mainScene = Scene()

    glfw.MakeContextCurrent(window.window)
    while not glfw.WindowShouldClose(window.window):
        # Render here
        # mainScene.renderModels(window.playerCamera)
        mainRender.renderOpenGL(window.playerCamera)

        # Swap front and back buffers
        glfw.SwapBuffers(window.window)

        # Poll for and process events
        glfw.PollEvents()

        processCameraSelectInput(window, parkingLot)

        if not window.cameraLocked:
            window.currentCamera.processPlayerInput(window.window)

    glfw.DestroyWindow(window.window)
    glfw.Terminate()

if __name__ == '__main__':
    # renderTest()
    main()
