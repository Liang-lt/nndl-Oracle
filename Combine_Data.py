import os
from PIL import Image
import matplotlib.pyplot as plt


def save_img(folder, img, new_img_path, fd_idx, img_idx):
    image = Image.open(os.path.join(folder, img))
    if fd_idx < 10:
        if img_idx < 10:
            plt.imsave(new_img_path + '/00{}_0{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
        else:
            plt.imsave(new_img_path + '/00{}_{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
    elif fd_idx < 100:
        if img_idx < 10:
            plt.imsave(new_img_path + '/0{}_0{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
        else:
            plt.imsave(new_img_path + '/0{}_{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
    else:
        if img_idx < 10:
            plt.imsave(new_img_path + '/{}_0{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
        else:
            plt.imsave(new_img_path + '/{}_{}.jpg'.format(fd_idx, img_idx), image, cmap='gray')
    plt.close()


def combine(shot, combine_folder, datasource='origin', useset='train', fd_shot=3):
    sec_folder = combine_folder.split('_')[1]
    if datasource != 'origin':
        fd_shot = shot


    data_folder = '../oracle_data/oracle_fs_{}/img/oracle_200_{}_shot/{}'.format(datasource, fd_shot, useset)
    loc = -len(useset)
    if not os.path.exists(data_folder[:loc] + '/' + combine_folder):
        os.mkdir(data_folder[:loc] + '/' + combine_folder)

    all_folders = [os.path.join(data_folder, folder) for folder in os.listdir(data_folder) if folder != '.DS_Store']

    for fd_idx, folder in enumerate(all_folders):
        # print(os.listdir(folder))
        new_img_path = '{}{}/{}'.format(data_folder[:loc], combine_folder, folder[-1])
        if not os.path.exists(new_img_path):
            os.mkdir(new_img_path)
        for img_idx, img in enumerate(os.listdir(folder)):
            save_img(folder, img, new_img_path, fd_idx, img_idx)

        # folder = folder[:-7] + sec_folder + folder[-2:]
        # print(folder)
        # for img_idx, img in enumerate(os.listdir(folder)):
        #     save_img(folder, img, new_img_path, fd_idx, img_idx + shot)


# combine(shot=5, combine_folder='train_test')


# combine(shot=1, combine_folder='train_query', fd_shot=3)
# combine(shot=1, combine_folder='train_test', useset='test', fd_shot=3)
#
# combine(shot=1, combine_folder='train_query', fd_shot=5)
# combine(shot=1, combine_folder='train_test', useset='test', fd_shot=5)


combine(shot=1, combine_folder='train_query', fd_shot=3, datasource='entire')
combine(shot=1, combine_folder='train_test', useset='test', fd_shot=3, datasource='entire')

combine(shot=3, combine_folder='train_query', fd_shot=5, datasource='entire')
combine(shot=3, combine_folder='train_test', useset='test', fd_shot=5, datasource='entire')


combine(shot=5, combine_folder='train_query', fd_shot=5, datasource='entire')
combine(shot=5, combine_folder='train_test', useset='test', fd_shot=5, datasource='entire')



# combine(shot=1, combine_folder='train_query', fd_shot=3, datasource='npz')
# combine(shot=1, combine_folder='train_test', useset='test', fd_shot=3, datasource='npz')
#
# combine(shot=3, combine_folder='train_query', fd_shot=5, datasource='npz')
# combine(shot=3, combine_folder='train_test', useset='test', fd_shot=5, datasource='npz')
#
#
# combine(shot=5, combine_folder='train_query', fd_shot=5, datasource='npz')
# combine(shot=5, combine_folder='train_test', useset='test', fd_shot=5, datasource='npz')