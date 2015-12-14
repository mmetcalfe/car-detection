# coding=utf-8

import pyglfw.pyglfw as fw


class CbWindow(fw.Window):
    def __init__(self, *args, **kwargs):
        super(CbWindow, self).__init__(*args, **kwargs)

        self.set_key_callback(CbWindow.key_callback)
        self.set_char_callback(CbWindow.char_callback)
        self.set_scroll_callback(CbWindow.scroll_callback)
        self.set_mouse_button_callback(CbWindow.mouse_button_callback)
        self.set_cursor_enter_callback(CbWindow.cursor_enter_callback)
        self.set_cursor_pos_callback(CbWindow.cursor_pos_callback)
        self.set_window_size_callback(CbWindow.window_size_callback)
        self.set_window_pos_callback(CbWindow.window_pos_callback)
        self.set_window_close_callback(CbWindow.window_close_callback)
        self.set_window_refresh_callback(CbWindow.window_refresh_callback)
        self.set_window_focus_callback(CbWindow.window_focus_callback)
        self.set_window_iconify_callback(CbWindow.window_iconify_callback)
        self.set_framebuffer_size_callback(CbWindow.framebuffer_size_callback)

    def key_callback(self, key, scancode, action, mods):
        print(
            "keybrd: key=%s scancode=%s action=%s mods=%s" %
            (key, scancode, action, mods))

    def char_callback(self, char):
        print("unichr: char=%s" % char)

    def scroll_callback(self, off_x, off_y):
        print("scroll: x=%s y=%s" % (off_x, off_y))

    def mouse_button_callback(self, button, action, mods):
        print("button: button=%s action=%s mods=%s" % (button, action, mods))

    def cursor_enter_callback(self, status):
        print("cursor: status=%s" % status)

    def cursor_pos_callback(self, pos_x, pos_y):
        print("curpos: x=%s y=%s" % (pos_x, pos_y))

    def window_size_callback(self, wsz_w, wsz_h):
        print("window: w=%s h=%s" % (wsz_w, wsz_h))

    def window_pos_callback(self, pos_x, pos_y):
        print("window: x=%s y=%s" % (pos_x, pos_y))

    def window_close_callback(self):
        print("should: %s" % self.should_close)

    def window_refresh_callback(self):
        print("redraw")

    def window_focus_callback(self, status):
        print("active: status=%s" % status)

    def window_iconify_callback(self, status):
        print("hidden: status=%s" % status)

    def framebuffer_size_callback(self, fbs_x, fbs_y):
        print("buffer: x=%s y=%s" % (fbs_x, fbs_y))


change_markers = {fw.Monitor.CONNECTED: '+', fw.Monitor.DISCONNECTED: '-'}


def on_monitor(_monitor, _event):
    change = change_markers.get(_event, '~')
    print("screen: %s %s" % (change, _monitor.name))


def main():
    fw.init()

    fw.Monitor.set_callback(on_monitor)

    win = CbWindow(800, 600, "callback window")
    win.make_current()

    while not win.should_close:

        win.swap_buffers()
        fw.poll_events()

        if win.keys.escape:
            win.should_close = True

    fw.terminate()

if __name__ == '__main__':
    main()
