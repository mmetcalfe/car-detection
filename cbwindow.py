import cyglfw3 as glfw
from OpenGL.GL import *
import numpy as np

from playercamera import PlayerCamera

class CbWindow:
    def __init__(self, width, height, title):
        glfw.WindowHint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.WindowHint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.WindowHint(glfw.OPENGL_FORWARD_COMPAT, 1)
        glfw.WindowHint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.WindowHint(glfw.RESIZABLE, True)
        glfw.WindowHint(glfw.RED_BITS, 8)
        glfw.WindowHint(glfw.GREEN_BITS, 8)
        glfw.WindowHint(glfw.BLUE_BITS, 8)
        # glfw.WindowHint(glfw.ALPHA_BITS, 8)
        glfw.WindowHint(glfw.DEPTH_BITS, 24)
        glfw.WindowHint(glfw.STENCIL_BITS, 8)

        self.window = glfw.CreateWindow(width, height, title)
        self.framebufferSize = (width, height)

        f = 240
        pos = np.array([3, 3, 3])
        dir = np.array([-1, -1, -1])
        up = np.array([0, 0, 1])
        near = 0.2
        far = 100
        self.playerCamera = PlayerCamera(f, self.framebufferSize, pos, dir, up, near, far)
        self.currentCamera = self.playerCamera
        self.cameraLocked = True
        self.mainRender = True

        def key_callback(window, key, scancode, action, mods):
            if (key == glfw.KEY_ESCAPE and action == glfw.PRESS):
                glfw.SetWindowShouldClose(window, True)

            if (action == glfw.PRESS and key == glfw.KEY_L):
                self.cameraLocked = not self.cameraLocked
                print 'cameraLocked:', self.cameraLocked

                glfw.SetInputMode(window, glfw.CURSOR, glfw.CURSOR_NORMAL if self.cameraLocked else glfw.CURSOR_DISABLED)

                # if not self.cameraLocked:
                # Ensure that locking/unlocking doesn't move the view:
                windowWidth, windowHeight = glfw.GetWindowSize(window)
                glfw.SetCursorPos(window, windowWidth / 2, windowHeight / 2)
                self.currentCamera.lastCursor = np.array(glfw.GetCursorPos(window), np.float32)
                self.currentCamera.lastUpdateTime = glfw.GetTime()

            pass
            # # print(
                # "keybrd: key=%s scancode=%s action=%s mods=%s" %
                # (key, scancode, action, mods))

        def char_callback(window, char):
            pass
            # print("unichr: char=%s" % char)

        def scroll_callback(window, off_x, off_y):
            pass
            # print("scroll: x=%s y=%s" % (off_x, off_y))

        def mouse_button_callback(window, button, action, mods):
            pass
            # print("button: button=%s action=%s mods=%s" % (button, action, mods))

        def cursor_enter_callback(window, status):
            pass
            # print("cursor: status=%s" % status)

        def cursor_pos_callback(window, pos_x, pos_y):
            pass
            # print("curpos: x=%s y=%s" % (pos_x, pos_y))

        def window_size_callback(window, wsz_w, wsz_h):
            pass
            # print("window: w=%s h=%s" % (wsz_w, wsz_h))

        def window_pos_callback(window, pos_x, pos_y):
            pass
            # print("window: x=%s y=%s" % (pos_x, pos_y))

        def window_close_callback(window):
            pass
            # print("should: %s" % self.should_close)

        def window_refresh_callback(window):
            pass
            # print("redraw")

        def window_focus_callback(window, status):
            pass
            # print("active: status=%s" % status)

        def window_iconify_callback(window, status):
            pass
            # print("hidden: status=%s" % status)

        def framebuffer_size_callback(window, fbs_x, fbs_y):
            print("buffer: x=%s y=%s" % (fbs_x, fbs_y))
            self.framebufferSize = (fbs_x, fbs_y)
            self.playerCamera.framebufferSize = (fbs_x, fbs_y)
            glViewport(0, 0, self.currentCamera.framebufferSize[0], self.currentCamera.framebufferSize[1])
            pass

        glfw.SetKeyCallback(self.window, key_callback)
        glfw.SetCharCallback(self.window, char_callback)
        glfw.SetScrollCallback(self.window, scroll_callback)
        glfw.SetMouseButtonCallback(self.window, mouse_button_callback)
        glfw.SetCursorEnterCallback(self.window, cursor_enter_callback)
        glfw.SetCursorPosCallback(self.window, cursor_pos_callback)
        glfw.SetWindowSizeCallback(self.window, window_size_callback)
        glfw.SetWindowPosCallback(self.window, window_pos_callback)
        glfw.SetWindowCloseCallback(self.window, window_close_callback)
        glfw.SetWindowRefreshCallback(self.window, window_refresh_callback)
        glfw.SetWindowFocusCallback(self.window, window_focus_callback)
        glfw.SetWindowIconifyCallback(self.window, window_iconify_callback)
        glfw.SetFramebufferSizeCallback(self.window, framebuffer_size_callback)
