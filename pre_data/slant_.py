

import numpy
import glob
from PIL import Image

class Slant():
    def __init__(self, complexity=1):
        # ---------- private attributes
        self.direction = 1
        self.angle = 0

        # ---------- generation parameters
        self.regenerate_parameters(complexity)
        # ------------------------------------------------

    def _get_current_parameters(self):
        return [self.angle, self.direction]

    def get_settings_names(self):
        return ['angle', 'direction']

    def regenerate_parameters(self, complexity):
        self.angle = numpy.random.uniform(0.0, complexity)
        P = numpy.random.uniform()
        self.direction = 1;
        if P < 0.5:
            self.direction = -1;
        return self._get_current_parameters()

    def transform_image(self, image):
        if self.angle == 0:
            return image

        ysize, xsize = image.shape
        slant = self.direction * self.angle

        output = image.copy()

        # shift all the rows
        for i in range(ysize):
            line = image[i]
            delta = round((i * slant)) % xsize
            line1 = line[:xsize - delta]
            line2 = line[xsize - delta:xsize]

            output[i][delta:xsize] = line1
            output[i][0:delta] = line2

        # correction to center the image
        correction = (self.direction) * round(self.angle * ysize / 2)
        correction = (xsize - correction) % xsize

        # center the region
        line1 = output[0:ysize, 0:xsize - correction].copy()
        line2 = output[0:ysize, xsize - correction:xsize].copy()
        output[0:ysize, correction:xsize] = line1
        output[0:ysize, 0:correction] = line2

        return output


# Test function
# Load an image in local and create several samples of the effect on the
# original image with different parameter. All the samples are saved in a single image, the 1st image being the original.

def test_slant(img_name, char_):
    # img_name = "test_img/下.jpg"
    root_dir = image_file[:image_file.rfind('/')]
    nb_samples = 5
    im = Image.open(img_name)
    im = im.convert("L")
    image = numpy.asarray(im)

    # image_final = image
    slant = Slant()
    im_list = []
    for i in range(nb_samples):
        dest_img_name = root_dir + "/{}_slanted{}.jpg".format(char_, i)
        slant.regenerate_parameters(1)
        image_slant = slant.transform_image(image)
        # image_final = numpy.hstack((image_final, image_slant))

        im = Image.fromarray(image_slant.astype('uint8'), "L")
        # im.save(dest_img_name)
        im_list.append(im)
    return im_list


def slant_(im):
    # # img_name = "test_img/下.jpg"
    # root_dir = image_file[:image_file.rfind('/')]
    nb_samples = numpy.random.randint(1, 5)
    # im = Image.open(img_name)
    im = im.convert("L")
    image = numpy.asarray(im)

    # image_final = image
    slant = Slant()
    for i in range(nb_samples):
        # dest_img_name = root_dir + "/{}_slanted{}.jpg".format(char_, i)
        slant.regenerate_parameters(1)
        image_slant = slant.transform_image(image)
        # image_final = numpy.hstack((image_final, image_slant))

        im = Image.fromarray(image_slant.astype('uint8'), "L")
        # im.save(dest_img_name)
    return im


if __name__ == '__main__':
    import sys, os, fnmatch


    image_path = 'oracle_fs/img/oracle_200_1_shot/train/*/'

    all_images = glob.glob(image_path + '*')

    print(len(all_images))
    i = 0

    for image_file in all_images:
        # print(image_file)
        char_ = image_file[image_file.rfind('/') - 1]
        test_slant(image_file, char_)

