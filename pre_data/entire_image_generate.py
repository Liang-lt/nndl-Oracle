import numpy as np
from slant_ import slant_
from dilation_ import dilate_
from erosion_ import erose_
from occlusion import occlu_
from pinch_ import pinch_
from perspective import perspective_
from affine_ import affine_

from PIL import Image
import glob
import os
import cv2


def entire_generate(image_path, gen_num=20):
    all_images = glob.glob(image_path + '*')
    for image_file in all_images:

        if image_file.find('entire_') != -1:
            # print(image_file)
            continue
        char_id = image_file.rfind('/')
        char = image_file[char_id - 1]
        # im.save(image_file[:-4] + '.jpg')

        # print(image_file)
        im = Image.open(image_file)
        for i in range(gen_num):
            flag = True
            while flag:
                img = cv2.cvtColor(np.asarray(im), cv2.COLOR_GRAY2BGR)
                if np.random.rand() < 0.15:
                    flag = False
                    img = dilate_(img)
                if np.random.rand() < 0.15:
                    flag = False
                    img = erose_(img)
                if np.random.rand() < 0.15:
                    flag = False
                    img = perspective_(img)
                if np.random.rand() < 0.15:
                    flag = False
                    img = pinch_(img)
                if np.random.rand() < 0.15:
                    flag = False
                    img = affine_(img)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

                if np.random.rand() < 0.3:
                    flag = False
                    img = slant_(img)
                if np.random.rand() < 0.15:
                    flag = False
                    img = occlu_(img)
            # print(image_file[:char_id + 1] + '{}_entire_{}.jpg'.format(char, i))
            img.save(image_file[:-4] + '{}_entire_{}.jpg'.format(char, i))


if __name__ == '__main__':
    np.random.seed(2022)
    image_path = '../oracle_data/oracle_fs_entire/img/*/train/'
    entire_generate(image_path)
