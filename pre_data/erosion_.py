

import cv2
import numpy as np



def erose_(img, dst=None):
    d = np.random.randint(1, 3, 2)
    # print(d)
    kernel = np.ones(d, np.uint8)
    dst1 = cv2.dilate(img, kernel, iterations=1)
    # if dst is not None:
    #     dst1 = np.hstack([dst, dst1])
    # cv.imshow('src', im_cv)
    return dst1


if __name__ == "__main__":
    img = cv2.imread('test_img/下.jpg')
    np.random.seed(123)
    dst = None
    for i in range(10):
        dst = erosion_(img, dst)

    cv2.imshow('dst', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()