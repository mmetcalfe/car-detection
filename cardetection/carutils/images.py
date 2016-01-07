import glob

def listImagesInDirectory(image_dir):
    image_list = glob.glob("{}/*.jpg".format(image_dir))
    image_list += glob.glob("{}/*.png".format(image_dir))
    return image_list
