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

    parkingLot = ParkingLot(
        size = np.array([5.0, 3.0]),
        centre = np.array([2.5, 1.5]),
        canvasSize = np.array([500, 300])
    )
    parkingLot.spaces.append(ParkingSpace([2, 0.5, 0], [0.8, 1.8]))
    parkingLot.spaces.append(ParkingSpace([1, 0.5, 0], [0.8, 1.8]))
    parkingLot.spaces.append(ParkingSpace([0, 0.5, 0], [0.8, 1.8]))
    mainRender = ParkingLotRender(parkingLot)

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

        if not window.cameraLocked:
            window.playerCamera.processPlayerInput(window.window)

    glfw.DestroyWindow(window.window)
    glfw.Terminate()

if __name__ == '__main__':
    # renderTest()
    main()
