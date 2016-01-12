# Loosely based on code from:
# http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import os.path
import cv2
from cardetection.carutils.images import listImagesInDirectory
import cardetection.carutils.geometry as gm
import cardetection.detection.cascadetraining as cascadetraining

def annotate_mouse_callback(event, x, y, flags, annotator):
    winName = annotator.winName

    # If adding a bounding box:
    if annotator.editing:
        # Set top left corner of rectangle:
        if event == cv2.EVENT_LBUTTONDOWN:
            annotator.rect_tl = (x, y)

        # Set bottom right corner of rectangle:
        elif event == cv2.EVENT_RBUTTONDOWN:
            annotator.rect_br = (x, y)

    # If deleting a bounding box:
    elif annotator.deleting:
        if event == cv2.EVENT_LBUTTONDOWN:
            print 'Delete at {}'.format((x, y))
            annotator.delete_rectangles_at_point((x, y))
            annotator.deleting = False

    annotator.update_display()

class OpenCVAnnotator(object):
    def __init__(self):
        self.winName = 'Annotator:{}'.format(id(self))

        # bbinfo_map :: Map Path [PixelRectangle]
        self.bbinfo_map = {}
        cv2.namedWindow(self.winName)

    def delete_rectangles_at_point(self, pt):
        rects = self.getImageRectangles(self.current_path)

        def rectContains(r, pt):
            h, w = self.current_img.shape[:2]
            oh, ow = self.original_shape[:2]
            r = r.scaleImage((ow, oh), (w,h))
            return r.contains(pt)

        rects = filter(lambda r: not rectContains(r, pt), rects)

        _, key = os.path.split(self.current_path)
        self.bbinfo_map[key] = rects

    def loadOpenCVBoundingBoxInfo(self, bbinfo_file):
        self.bbinfo_map = {}
        bbinfo_cache = cascadetraining.loadInfoFile(bbinfo_file)
        for k in bbinfo_cache:
            self.bbinfo_map[k] = cascadetraining.rectanglesFromCacheString(bbinfo_cache[k])

    def saveOpenCVBoundingBoxInfo(self, bbinfo_file):
        # Convert the bbinfo_map into a list of bbinfo_lines:
        bbinfo_lines = []
        for img_path, bboxes in self.bbinfo_map.iteritems():
            num = len(bboxes)
            if num == 0:
                continue
            bbox_strings = [' '.join(map(str, b.opencv_bbox)) for b in bboxes]
            bbinfo_line = '{} {} {}'.format(img_path, num, ' '.join(bbox_strings))
            bbinfo_lines.append(bbinfo_line)

        # Write the bbinfo_lines to the bbinfo_file:
        with open(bbinfo_file, 'w') as fh:
            fh.write('\n'.join(bbinfo_lines))

    def addRectangleToImage(self, img_path, rect):
        _, key = os.path.split(img_path)

        if not key in self.bbinfo_map:
            self.bbinfo_map[key] = []

        h, w = self.current_img.shape[:2]
        if self.flipped:
            rect = rect.flipped((w,h))

        oh, ow = self.original_shape[:2]
        rect = rect.scaleImage((w,h), (ow, oh))

        self.bbinfo_map[key].append(rect)

    def getImageRectangles(self, img_path):
        _, key = os.path.split(img_path)

        if not key in self.bbinfo_map:
            return []

        return self.bbinfo_map[key]

    # Annotate all images in the directory:
    def annotate_directory(self, img_dir, bbinfo_file):
        # Get sorted image list:
        img_paths = sorted(listImagesInDirectory(img_dir))

        self.img_index = 0 # Set initial index.
        self.current_path = img_paths[self.img_index]
        self.current_img = cv2.imread(self.current_path)
        self.original_shape = self.current_img.shape
        self.current_scale = 1.0
        self.flipped = False

        self.deleting = False
        self.editing = False
        self.rect_tl = None
        self.rect_br = None

        if os.path.isfile(bbinfo_file):
            self.loadOpenCVBoundingBoxInfo(bbinfo_file)

        cv2.setMouseCallback(self.winName, annotate_mouse_callback, self)
        while True:
            self.update_display()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print 'Quitting'
                break

            # Save all bounding boxes to the output file:
            elif key == ord('s'):
                print 'Saved: {}'.format(bbinfo_file)
                self.saveOpenCVBoundingBoxInfo(bbinfo_file)

            # Overwrite current image with a rotated copy:
            elif key == ord('i'):
                if self.flipped:
                    print 'Overwritten with flipped image: {}'.format(bbinfo_file)
                    self.current_img = cv2.imread(self.current_path)
                    self.current_img = cv2.flip(self.current_img, -1)
                    cv2.imwrite(self.current_path, self.current_img)
                    self.flipped = False
                    h, w = self.current_img.shape[:2]
                    rects = self.getImageRectangles(self.current_path)
                    rects = map(lambda r: r.flipped((w,h)), rects)
                    _, key = os.path.split(self.current_path)
                    self.bbinfo_map[key] = rects
                else:
                    print 'Current image has not been rotated.'

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
                    rect = gm.PixelRectangle.fromCorners(self.rect_tl, self.rect_br)
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
                self.original_shape = self.current_img.shape[:2]
                self.flipped = False
                print 'Moved to image: {}, {}'.format(self.img_index, self.current_path)

            elif (key == 30 or key == 0): # up arrow
                self.current_img = cv2.flip(self.current_img, -1)
                self.flipped = not self.flipped

                if self.rect_tl and self.rect_br:
                    h, w = self.current_img.shape[:2]
                    rect = gm.PixelRectangle.fromCorners(self.rect_tl, self.rect_br)
                    rect = rect.flipped((w,h))
                    self.rect_tl = tuple(rect.tl)
                    self.rect_br = tuple(rect.br)
                print 'Flipped image.'

            # ASCII DEL: 127 (the backspace key on OSX)
            # What OpenCV makes of my delete key: 40
            elif (key == 127 or key == 40):
                self.deleting = True
                print 'Click a bounding box to DELETE it.'

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

        # Draw bounding boxes for current image:
        for rect in self.getImageRectangles(self.current_path):
            oh, ow = self.original_shape[:2]
            rect = rect.scaleImage((ow, oh), (w,h))
            if self.flipped:
                rect = rect.flipped((w,h))

            tl = tuple(rect.tl)
            br = tuple(rect.br)
            cv2.rectangle(clone_img, tl, br, (255, 255, 255), 3)
            cv2.rectangle(clone_img, tl, br, (255, 0, 127), 2)

        # Draw bounding box being edited:
        if self.rect_tl and self.rect_br:
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
