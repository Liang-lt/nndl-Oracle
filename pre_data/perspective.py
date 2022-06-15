import cv2
import numpy as np


def perspective_(img, dst=None):
    d = list(np.random.randint(0, 8, 8))
    # print(d)
    # kernel = np.ones(d, np.uint8)
    # dst1 = cv2.erode(img, kernel, iterations=1)
    # if dst is not None:
    #     dst1 = np.hstack([dst, dst1])
    # cv.imshow('src', im_cv)
    w, h = img.shape[0], img.shape[1]
    points1 = np.float32([(0, 0), (w, 0), (0, h), (w, h)])
    points2 = np.float32([(d[0], d[1]), (w-d[2], d[3]), (d[4], h-d[5]), (w-d[6], h-d[7])])

    M = cv2.getPerspectiveTransform(points1, points2)

    dst1 = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))
    # if dst is not None:
    #     dst1 = np.hstack([dst, dst1])
    return dst1


if __name__ == '__main__':
    img = cv2.imread('test_img/下.jpg')
    # np.random.seed(123)
    dst = img
    for i in range(10):
        dst = perspective_(img, dst)

    cv2.imshow('dst', dst)

    cv2.waitKey(0)
    cv2.destroyAllWindows()