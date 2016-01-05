# coding=utf-8

import pyglfw.pyglfw as glfw

from OpenGL.GL import *
from math import *

import window


class Render(object):
    def __init__(self, viewport):
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def __call__(self, *args):
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        for _obj in args:
            _obj()

    def triangle(self, center, length=0.1):
        def draw():
            glBegin(GL_TRIANGLES)

            glVertex2f(center[0], center[1])
            glVertex2f(center[0] - length, center[1] + length)
            glVertex2f(center[0] + length, center[1] + length)

            glEnd()

        return draw

    def quad(self, center, length=0.1):
        def draw():
            glBegin(GL_QUADS)

            glVertex2f(center[0] - length, center[1] - length)
            glVertex2f(center[0] - length, center[1] + length)
            glVertex2f(center[0] + length, center[1] + length)
            glVertex2f(center[0] + length, center[1] - length)

            glEnd()

        return draw


class Domain(object):
    def __init__(self):
        self.cnt_x = 0
        self.cnt_y = 0

        self.pos_x = 0
        self.pos_y = 0

    @property
    def points(self):
        return [(self.pos_x, self.pos_y)]

    def mov(self, vel_x, vel_y):
        self.pos_x += vel_x
        self.pos_y += vel_y

    @property
    def pos(self):
        return (self.pos_x, self.pos_y)

    @pos.setter
    def pos(self, x_y):
        x, y = x_y
        self.pos_x = self.cnt_x + x
        self.pos_y = self.cnt_y + y


# def on_key(window, key, scancode, action, mods):
#     if action == glfw.Window.PRESS and key == glfw.Keys.ESCAPE:
#         window.should_close = True

if __name__ == '__main__':

    glfw.init()

    # pm = glfw.get_primary_monitor()
    # vm = pm.video_modes[-1]
    # win = glfw.Window(vm.width, vm.height, "nayadra", pm)

    # win = glfw.Window(800, 600, "nayadra")

    win = window.CbWindow(800, 600, "nayadra")

    # TODO: Verify that some of these work.
    # NOTE: resizable does not work.
    win.hint(
        context_ver_major = 3,
        context_ver_minor = 3,
        opengl_profile = glfw.api.GLFW_OPENGL_CORE_PROFILE,
        opengl_forward_compat = True,
        resizable = False,
        red_bits = 8,
        green_bits = 8,
        blue_bits = 8,
    #    alpha_bits = 8,
        depth_bits = 24,
        stencil_bits = 8
    )

    win.make_current()
    win.swap_interval(0) # Disable v-sync
    # win.set_key_callback(on_key)



    # if not win.monitor == pm:
    #     raise Exception("Wrong monitor set!")

    jst = glfw.Joystick(0)

    with win:
        render = Render(win.framebuffer_size)

    dom = Domain()

    def calc_movement(jst, n_axis, keyneg, keypos):

        jst_move = jst and round(jst.axes[n_axis], 1) and jst.axes[n_axis]

        return (jst_move or (float(keyneg) - float(keypos)))

    while not win.should_close:
        glfw.poll_events()

        if win.keys.ESCAPE:
            window.should_close = True

        mov_x = calc_movement(jst, 0, win.keys.right, win.keys.left)
        mov_y = calc_movement(jst, 1, win.keys.up, win.keys.down)

        dom.mov(mov_x * 0.01, mov_y * 0.01)

#        dom.pos = jst.axes[0], jst.axes[1]

        with win:
            drawes = [render.quad(p) for p in dom.points]
            render(*drawes)

        win.swap_buffers()

    glfw.terminate()
