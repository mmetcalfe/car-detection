# Loosely based on code from:
# http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import cv2
from cardetection.carutils.images import listImagesInDirectory
import cardetection.carutils.geometry as gm

def annotate_mouse_callback(event, x, y, flags, annotator):
    winName = annotator.winName

    # Set top left corner of rectangle:
    if annotator.editing:
        if event == cv2.EVENT_LBUTTONDOWN:
            annotator.rect_tl = (x, y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            annotator.rect_br = (x, y)

    annotator.update_display()

class OpenCVAnnotator(object):
    def __init__(self):
        self.winName = 'Annotator:{}'.format(id(self))
        self.bbinfo = {}
        cv2.namedWindow(self.winName)

    def addRectangleToImage(self, img_path, rect):
        if not img_path in self.bbinfo:
            self.bbinfo[img_path] = []

        if self.flipped:
            h, w = self.current_img.shape[:2]
            rect = rect.flipped((w,h))

        self.bbinfo[img_path].append(rect)

    def getImageRectangles(self, img_path):
        if not img_path in self.bbinfo:
            return []

        return self.bbinfo[img_path]

    # Annotate all images in the directory:
    def annotate_directory(self, img_dir, bbinfo_file):
        # Get sorted image list:
        img_paths = sorted(listImagesInDirectory(img_dir))

        self.img_index = 0 # Set initial index.
        self.current_path = img_paths[self.img_index]
        self.current_img = cv2.imread(self.current_path)
        self.current_scale = 1.0
        self.flipped = False

        self.editing = False
        self.rect_tl = None
        self.rect_br = None

        cv2.setMouseCallback(self.winName, annotate_mouse_callback, self)
        while True:
            self.update_display()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Add a new bounding box:
            elif key == ord('n'):
                print 'New bounding box'
                self.editing = True

            # Cancel:
            elif key == 27: # ESC key
                print 'Cancel'
                self.editing = False

            # Accept bounding box:
            elif key == 13: # Enter / CR key
                if not self.rect_tl is None and not self.rect_br is None:
                    rect = gm.PixelRetangle.fromCorners(self.rect_tl, self.rect_br)
                    self.addRectangleToImage(self.current_path, rect)
                    print 'Bounding box added'
                    self.editing = False
                    self.rect_tl = None
                    self.rect_br = None
                else:
                    print 'WARNING: Invalid bounding box was not added'

            # ASCII Codes:
            #   {28: 'right arrow', 29: 'left arrow'}
            #   {30: 'up arrow', 31: 'down arrow'}
            # Non-standard codes that work for me:
            #   r: 3, l: 2, u: 0, d: 1
            # Move to next/previous image:
            elif (key == 28 or key == 3) or (key == 29 or key == 2):
                self.img_index += 1 if (key == 28 or key == 3) else -1
                self.img_index %= len(img_paths)
                self.current_path = img_paths[self.img_index]
                self.current_img = cv2.imread(self.current_path)
                self.flipped = False
                print 'Moved to image: {}, {}'.format(self.img_index, self.current_path)

            elif (key == 30 or key == 0): # up arrow
                self.current_img = cv2.flip(self.current_img, -1)
                self.flipped = not self.flipped
                print 'Flipped image.'

            elif key != 255: # 255 means no key was pressed
                print 'Unused keycode: {}'.format(key)

    def update_display(self):
        h, w = self.current_img.shape[:2]
        max_dim = 1024.0
        if w > max_dim + 50 or h > max_dim + 50:
            sx = max_dim / w
            sy = max_dim / h
            self.current_scale = min(sx, sy)
            self.current_img = cv2.resize(self.current_img, dsize=None, fx=self.current_scale, fy=self.current_scale)

        clone_img = self.current_img.copy()

        for rect in self.getImageRectangles(self.current_path):
            if self.flipped:
                rect = rect.flipped((w,h))
            tl = tuple(rect.tl)
            br = tuple(rect.br)
            cv2.rectangle(clone_img, tl, br, (255, 255, 255), 3)
            cv2.rectangle(clone_img, tl, br, (255, 0, 127), 2)

        if not self.rect_tl is None and not self.rect_br is None:
            cv2.rectangle(clone_img, self.rect_tl, self.rect_br, (255, 255, 255), 3)
            cv2.rectangle(clone_img, self.rect_tl, self.rect_br, (0, 255, 0), 2)

        cv2.imshow(self.winName, clone_img)
        # cv2.imshow(self.winName, self.current_img)

    def destroy(self):
        cv2.destroyWindow(self.winName)

if __name__ == '__main__':
    img_dir = '/Users/mitchell/data/car-detection/shopping'
    bbinfo_file = '/Users/mitchell/data/car-detection/bbinfo/shopping__bbinfo.dat'

    annotator = OpenCVAnnotator()
    annotator.annotate_directory(img_dir, bbinfo_file)
    annotator.destroy()
