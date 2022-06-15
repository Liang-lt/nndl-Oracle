import cv2
import numpy as np


def affine_(img, dst=None):
    d = np.random.randint(-3, 3, 2)
    # print(d)
    # kernel = np.ones(d, np.uint8)
    # dst1 = cv2.erode(img, kernel, iterations=1)
    # if dst is not None:
    #     dst1 = np.hstack([dst, dst1])
    # cv.imshow('src', im_cv)

    M = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
    dst1 = cv2.warpAffine(img, M, (img.shape[0], img.shape[1]), borderMode=cv2.BORDER_REFLECT)
    # if dst is not None:
    #     dst1 = np.hstack([dst, dst1])
    return dst1


if __name__ == '__main__':
    img = cv2.imread('test_img/下.jpg')
    # np.random.seed(123)
    dst = img
    for i in range(10):
        dst = affine_(img, dst)

    cv2.imshow('dst', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()