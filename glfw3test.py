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
from render import Scene

def main():
    if not glfw.Init():
        exit()

    window = CbWindow(640, 480, 'window')

    if not window.window:
        glfw.Terminate()
        print('GLFW: Failed to create window')
        exit()

    print glGetString(GL_VERSION)

    mainScene = Scene()

    glfw.MakeContextCurrent(window.window)
    while not glfw.WindowShouldClose(window.window):
        # Render here
        mainScene.renderModels(window.playerCamera)

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
