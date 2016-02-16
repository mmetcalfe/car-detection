# Loosely based on code from:
# http://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/

import argparse
import os.path
import cv2
import cardetection.carutils.images as utils
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
        rects = self.get_image_rectangles(self.current_path)

        def rectContains(r, pt):
            h, w = self.current_img.shape[:2]
            oh, ow = self.original_shape[:2]
            r = r.scale_image((ow, oh), (w,h))
            return r.contains(pt)

        rects = filter(lambda r: not rectContains(r, pt), rects)

        _, key = os.path.split(self.current_path)
        self.bbinfo_map[key] = rects

    def load_opencv_bounding_box_info(self, bbinfo_file):
        self.bbinfo_map = utils.load_opencv_bounding_box_info(bbinfo_file)

    def save_opencv_bounding_box_info(self, bbinfo_file):
        utils.save_opencv_bounding_box_info(bbinfo_file, self.bbinfo_map)

    def convert_to_rect_in_original(self, rect):
        h, w = self.current_img.shape[:2]
        if self.flipped:
            rect = rect.rotated_180((w,h))

        oh, ow = self.original_shape[:2]
        rect = rect.scale_image((w,h), (ow, oh))
        return rect

    def add_rectangle_to_image(self, img_path, rect):
        _, key = os.path.split(img_path)

        if not key in self.bbinfo_map:
            self.bbinfo_map[key] = []

        rect = self.convert_to_rect_in_original(rect)

        self.bbinfo_map[key].append(rect)

    def get_image_rectangles(self, img_path):
        _, key = os.path.split(img_path)

        if not key in self.bbinfo_map:
            return []

        return self.bbinfo_map[key]

    def get_annotation_count(self):
        total = 0
        for img_path, bboxes in self.bbinfo_map.iteritems():
            num = len(bboxes)
            total += num
        return total

    # Annotate all images in the directory:
    def annotate_directory(self, img_dir, bbinfo_file, display_width):
        self.display_width = display_width

        # Get sorted image list:
        img_paths = sorted(utils.list_images_in_directory(img_dir))

        self.img_index = 0 # Set initial index.
        self.current_path = img_paths[self.img_index]
        self.current_img = cv2.imread(self.current_path)
        self.original_img = self.current_img
        self.original_shape = self.current_img.shape
        self.current_scale = 1.0
        self.flipped = False

        self.deleting = False
        self.editing = False
        self.rect_tl = None
        self.rect_br = None

        if os.path.isfile(bbinfo_file):
            self.load_opencv_bounding_box_info(bbinfo_file)

        cv2.setMouseCallback(self.winName, annotate_mouse_callback, self)
        while True:
            self.update_display()
            key = cv2.waitKey(1) & 0xFF

            # Use arrow keys, or < and > to move between images.
            # ASCII Codes:
            #   {28: 'right arrow', 29: 'left arrow'}
            #   {30: 'up arrow', 31: 'down arrow'}
            # Non-standard codes that work for me:
            #   r: 3, l: 2, u: 0, d: 1
            right_pressed = (key == 28 or key == 46 or key == 3)
            left_pressed = (key == 29 or key == 44 or key == 2)

            # Use delete or backspace to enter delete mode.
            # ASCII DEL: 127 (the backspace key on OSX)
            # What OpenCV makes of my delete key: 40
            delete_pressed = (key == 127 or key == 40 or key == 8)

            # Use up arrow or 'f' to flip an image.
            up_pressed = (key == 30 or key == 0)
            flip_pressed = up_pressed or key == ord('f')

            # Space pressed:
            space_pressed = key == 32

            if key == ord('q'):
                print 'Quitting'
                break

            # Save all bounding boxes to the output file:
            elif key == ord('s'):
                print 'Saved: {}'.format(bbinfo_file)
                self.save_opencv_bounding_box_info(bbinfo_file)

            # Overwrite current image with a rotated copy:
            elif key == ord('i'):
                if self.flipped:
                    print 'Overwritten with flipped image: {}'.format(bbinfo_file)
                    self.current_img = cv2.imread(self.current_path)
                    self.current_img = cv2.flip(self.current_img, -1)
                    cv2.imwrite(self.current_path, self.current_img)
                    self.original_img = self.current_img
                    self.flipped = False
                    h, w = self.current_img.shape[:2]
                    rects = self.get_image_rectangles(self.current_path)
                    rects = map(lambda r: r.rotated_180((w,h)), rects)
                    _, key = os.path.split(self.current_path)
                    self.bbinfo_map[key] = rects
                else:
                    print 'Current image has not been rotated.'

            # Add a new bounding box:
            elif key == ord('n') or (space_pressed and not self.editing):
                print 'New bounding box'
                self.editing = True

            # Cancel:
            elif key == 27: # ESC key
                print 'Cancel'
                self.editing = False

            # Accept bounding box:
            elif key == 13 or (space_pressed and self.editing): # Enter / CR key
                if not self.rect_tl is None and not self.rect_br is None:
                    rect = gm.PixelRectangle.fromCorners(self.rect_tl, self.rect_br)
                    if rect.w > 10 and rect.h > 10:
                        self.add_rectangle_to_image(self.current_path, rect)
                        print 'Bounding box {} added.'.format(self.get_annotation_count())
                        self.editing = False
                        self.rect_tl = None
                        self.rect_br = None
                    else:
                        print 'WARNING: Small ({}x{}) bounding box was not added'.format(rect.w, rect.h)
                else:
                    print 'WARNING: Invalid bounding box was not added'

            # Move to next/previous image:
            elif right_pressed or left_pressed:
                self.img_index += 1 if right_pressed else -1
                self.img_index %= len(img_paths)
                self.current_path = img_paths[self.img_index]
                self.current_img = cv2.imread(self.current_path)
                self.original_img = self.current_img
                self.original_shape = self.current_img.shape[:2]
                self.flipped = False
                print 'Moved to image: {}, {}'.format(self.img_index, self.current_path)

            elif flip_pressed: # up arrow
                self.current_img = cv2.flip(self.current_img, -1)
                self.original_img = cv2.flip(self.original_img, -1)
                self.flipped = not self.flipped

                if self.rect_tl and self.rect_br:
                    h, w = self.current_img.shape[:2]
                    rect = gm.PixelRectangle.fromCorners(self.rect_tl, self.rect_br)
                    rect = rect.rotated_180((w,h))
                    self.rect_tl = tuple(rect.tl)
                    self.rect_br = tuple(rect.br)
                print 'Flipped image.'

            elif delete_pressed:
                self.deleting = True
                print 'Click a bounding box to DELETE it.'

            elif key != 255: # 255 means no key was pressed
                print 'Unused keycode: {}'.format(key)

    def update_display(self):
        h, w = self.current_img.shape[:2]
        max_dim = float(self.display_width)
        if w > max_dim + 50 or h > max_dim + 50:
            sx = max_dim / w
            sy = max_dim / h
            self.current_scale = min(sx, sy)
            self.current_img = cv2.resize(self.current_img, dsize=None, fx=self.current_scale, fy=self.current_scale)

        clone_img = self.current_img.copy()

        # Draw bounding boxes for current image:
        for rect in self.get_image_rectangles(self.current_path):
            oh, ow = self.original_shape[:2]
            rect = rect.scale_image((ow, oh), (w,h))
            if self.flipped:
                rect = rect.rotated_180((w,h))

            tl = tuple(rect.tl)
            br = tuple(rect.br)
            cv2.rectangle(clone_img, tl, br, (255, 255, 255), 3)
            cv2.rectangle(clone_img, tl, br, (255, 0, 127), 2)

        # Draw bounding box being edited:
        if self.rect_tl and self.rect_br:
            cv2.rectangle(clone_img, self.rect_tl, self.rect_br, (255, 255, 255), 3)
            cv2.rectangle(clone_img, self.rect_tl, self.rect_br, (0, 255, 0), 2)

            rect = gm.PixelRectangle.fromCorners(self.rect_tl, self.rect_br)
            rect = self.convert_to_rect_in_original(rect)

            cropped = utils.crop_rectangle(self.original_img, rect)
            cw = self.current_img.shape[1]
            ch = self.current_img.shape[0]
            if rect.w > 5 and rect.h > 5:
                if rect.w > rect.h:
                    preview_width = cw / 5.0
                    preview_height = preview_width / rect.aspect
                else:
                    preview_height = ch / 4.0
                    preview_width = preview_height * rect.aspect
                new_shape = (max(2, int(round(preview_height))), max(2, int(round(preview_width))))
                preview = utils.resize_sample(cropped, new_shape, use_interp=False)
                clone_img[0:new_shape[0],0:new_shape[1],:] = preview

        cv2.imshow(self.winName, clone_img)
        # cv2.imshow(self.winName, self.current_img)

    def destroy(self):
        cv2.destroyWindow(self.winName)

if __name__ == '__main__':
    # Parse arguments:
    parser = argparse.ArgumentParser(description='Train a HOG + Linear SVM classifier.')
    parser.add_argument('img_dir', type=str, nargs='?', default='/Users/mitchell/data/car-detection/university', help='Directory containing the images to annotate.')
    parser.add_argument('info_file', type=str, nargs='?', default='/Users/mitchell/data/car-detection/bbinfo/university__bbinfo.dat', help='File to which annotation data will be saved.')
    parser.add_argument('display_width', type=int, nargs='?', default=1500, help='Width of the annotation window.')
    args = parser.parse_args()

    img_dir = args.img_dir
    bbinfo_file = args.info_file
    display_width = args.display_width
    # bbinfo_file = '/Users/mitchell/data/car-detection/bbinfo/university__exclusion.dat'

    img_dir = '/Users/mitchell/data/car-detection/university'
    bbinfo_file = '/Users/mitchell/data/car-detection/bbinfo/university__bbinfo.dat'
    print 'img_dir:', img_dir
    print 'bbinfo_file:', bbinfo_file
    print 'display_width:', repr(display_width)

    annotator = OpenCVAnnotator()
    annotator.annotate_directory(img_dir, bbinfo_file, display_width)
    annotator.destroy()
