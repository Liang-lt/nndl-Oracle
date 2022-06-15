
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob
from dilation_ import dilate_
from erosion_ import erose_
from affine_ import affine_
from perspective import perspective_
from occlusion import occlu_
from pinch_ import pinch_
from slant_ import slant_


def strokes_to_lines(strokes):
    """Convert stroke–3 format to polyline format."""
    x = 0
    y = 0
    lines = []
    line = []
    for i in range(len(strokes)):
        if strokes[i, 2] == 1:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
            lines.append(line)
            line = []
        else:
            x += float(strokes[i, 0])
            y += float(strokes[i, 1])
            line.append([x, y])
    return lines


def show_one_sample(strokes):
    lines = strokes_to_lines(strokes)
    fig = plt.figure(figsize=(6, 6))

    for idx in range(0, len(lines)):
        x = [x[0] for x in lines[idx]]
        y = [y[1] for y in lines[idx]]
        plt.plot(x, y, 'k-', linewidth=5)
    ax = plt.gca()
    plt.xticks([])
    plt.yticks([])
    # plt.axes().get_xaxis().set_visible(False)
    # plt.axes().get_yaxis().set_visible(False)
    # ax.xaxis.set_ticks_position('top')
    ax.invert_yaxis()
    # print()
    plt.axis('off')
    # plt.show()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # print(w, h)
    # 将Image.frombytes替换为Image.frombuffer,图像会倒置
    img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
    # img = img.crop((50, 50, 550, 550))
    # cv2.imshow(img)

    # print(img.size)
    print(img)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    # img = dilate_(img)
    kernel = np.ones((11, 11), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.resize((28, 28), resample=Image.LANCZOS)
    # img.show()


def txt2list(path):
    data_list = []
    with open(path, "r") as f:
        file = f.readlines()
        # print(file)
        for line in file:
            line = line.strip("\n")  # 去除末尾的换行符
            for c in line:
                data_list.append(c)
    return data_list


def npz_(npz_root, c_list, k_shot, gen_num=20):
    # npz_path = 'oracle_fs/seq/oracle_200_1_shot/199.npz'
    all_images = glob.glob(npz_root + '*')
    img_path = 'oracle_data/oracle_fs_npz/img/oracle_200_{}_shot/train/'.format(k_shot)
    for npz_path in all_images:
        if npz_path.find('txt') != -1:
            continue

        l = npz_path.rfind('/')
        r = npz_path.rfind('.')
        npz_id = npz_path[l+1:r]
        char = c_list[int(npz_id)]
        if char != '歳':
            continue
        print(npz_path)
        # print(char)
        data = np.load(npz_path, encoding='latin1', allow_pickle=True)
        train_data = data['train']
        if len(train_data) == 1:
            train_strokes = train_data[0]
            for i in range(gen_num):
                # print(i)
                img = stroke_step(train_strokes)
                while img is None:
                    img = stroke_step(train_strokes)
                img.save(img_path + '{}/{}_npz{}.jpg'.format(char, char, i))

            # print(char)
            # break
            # print(train_strokes)
        else:
            k = 0
            for train_strokes in train_data:
                for i in range(gen_num):
                    img = stroke_step(train_strokes)
                    while img is None:
                        img = stroke_step(train_strokes)
                    img.save(img_path + '{}/{}_npz{}_{}.jpg'.format(char, char, k, i))
                k += 1



def stroke_step(strokes):
    lines = strokes_to_lines(strokes)
    fig = plt.figure(figsize=(5, 10))
    img_list = []
    flag = False
    # print(len(lines))

    for idx in range(0, len(lines)):
        x = [x[0] for x in lines[idx]]
        y = [y[1] for y in lines[idx]]
        plt.plot(x, y, 'k-', linewidth=5)
        if np.random.rand() < 0.9 and idx < len(lines) - 1:
            continue
        else:
            flag = True
            if len(img_list) > 1:
                if idx < len(lines) - 1:
                    continue
                else:
                    img_final = img.convert('L')
            ax = plt.gca()
            plt.xticks([])
            plt.yticks([])
            # plt.axes().get_xaxis().set_visible(False)
            # plt.axes().get_yaxis().set_visible(False)
            # ax.xaxis.set_ticks_position('top')
            ax.invert_yaxis()
            # print()
            plt.axis('off')
            # plt.show()
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            # print(w, h)
            # 将Image.frombytes替换为Image.frombuffer,图像会倒置
            img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
            # img = img.crop((50, 50, 550, 550))
            # cv2.imshow(img)

            # print(img.size)
            p = np.random.rand()
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
            kernel = np.ones((11, 11), np.uint8)
            img = cv2.erode(img, kernel, iterations=1)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
            img = img.convert('L')
            img = img.resize((28, 28), resample=Image.LANCZOS)
            # img = img.convert('L')
            # print("before:", img.size)
            # print(p)
            if p < 0.6:
                img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                # img = dilate_(img)
                if p < 0.1:
                    img = dilate_(img)
                elif p < 0.2:
                    img = erose_(img)
                elif p < 0.3:
                    img = perspective_(img)
                elif p < 0.4:
                    img = pinch_(img)
                else:
                    img = affine_(img)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            else:
                if p < 0.7:
                    img = slant_(img)
                else:
                    img = occlu_(img)
            if idx == len(lines) - 1:
                img_final = img.convert('L')
            else:
                # img = dilate_(img)
                # kernel = np.ones((11, 11), np.uint8)
                # img = cv2.erode(img, kernel, iterations=1)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # img = img.resize((28, 28), resample=Image.LANCZOS)
                img_list.append(img)
                # print(np.array(img).)
                # plt.show()
                plt.close(fig)
                # plt.show()
                fig = plt.figure(figsize=(6, 6))
                # plt.show()

            # img.show()
    plt.close('all')
    if flag is not True:
        return None
    img_final = np.array(img_final)
    for im in img_list:
        im = np.array(im)
        img_final = get_min(img_final, im)
    img_final = Image.fromarray(img_final, 'L')
    # img_final.show()
    return img_final

def get_min(im1, im2):
    x, y = im1.shape
    # if im1.shape != im2.shape:
    #     raise Exception
    for i in range(x):
        for j in range(y):
            im1[i][j] = min(im1[i][j], im2[i][j])
    return im1




np.random.seed(2022)
k_shot = 1
txt_path = '../oracle_data/oracle_fs_npz/seq/char_to_idx.txt'
image_path = '../oracle_data/oracle_fs_npz/seq/oracle_200_{}_shot/'.format(k_shot)
c_list = txt2list(txt_path)
npz_(image_path, c_list, k_shot)
