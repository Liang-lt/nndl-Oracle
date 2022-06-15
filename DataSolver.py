#!/usr/bin/python3.6
""" Code for loading data. """

import torch
from torch.utils.data import Dataset
from PIL import Image

import numpy as np
import os
import random
from tqdm import tqdm
import pickle


def get_images(paths, num_classes, nb_samples=None, shuffle=True):
    labels = range(num_classes)
    rots = np.random.randint(num_classes, size=(num_classes))
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, np.rot90(Image.open(os.path.join(path, image)), rot)[np.newaxis, :]) \
              for i, rot, path in zip(labels, rots, paths) \
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


def get_files(folders, frange, num_classes, shot, query, fd_shot):
    labels = range(num_classes)
    rots = np.random.randint(4, size=num_classes)
    indexes = random.sample(frange, num_classes)
    random.shuffle(indexes)

    images = []
    # print(folders)
    # print(labels, rots, indexes)
    for i, rot, index in zip(labels, rots, indexes):
        folder_len = len(os.listdir(folders[index]))
        for idx in random.sample(range(folder_len), shot + query):
            images.append((i, rot, index * folder_len + idx))
    return images


class Oracle(Dataset):
    def __init__(self, use_set='train', resume_itr=0, config={}):
        """
        Args:
            k_shot: num samples to generate per class in one batch to train
            k_query: num samples to generate per class in one batch to test
        """
        self.images = []
        self.config = config
        self.use_set = use_set
        self.k_shot = config.get('k_shot')
        self.k_query = config.get('k_query')
        self.fd_shot = config.get('fd_shot', 5)
        self.img_num = config.get('img_num', 20)
        self.data_num = config.get('data_num', 1500)
        self.meta_batch_size = config.get('meta_batch_size', 1)
        self.resume_index = resume_itr * self.meta_batch_size
        self.resume_val_idx = ((resume_itr - 1) // config.get('save_iter', 1000) + 1) * self.meta_batch_size
        self.num_samples_per_class = self.k_shot + self.k_query
        self.num_classes = config.get('num_classes', 5)
        self.num_shot_per_task = self.k_shot * self.num_classes
        self.num_query_per_task = self.k_query * self.num_classes
        self.img_size = config.get('img_size', (28, 28))
        self.data_source = config.get('data_source', 'origin')
        if self.data_source != 'origin':
            self.fd_shot = self.k_shot

        if self.use_set == 'train':
            self.data_folder = 'oracle_data/oracle_fs_{}/img/oracle_200_{}_shot/train_query'.format(self.data_source, self.fd_shot)
            self.all_folders = [os.path.join(self.data_folder, folder) for folder in os.listdir(self.data_folder)]
            if self.data_source == 'origin':
                self.data_num = 10000
            elif self.data_source == 'entire':
                self.data_num = 20000
            elif self.data_source == 'npz':
                self.data_num = 40000
        elif self.use_set == 'test':
            self.data_folder = 'oracle_data/oracle_fs_{}/img/oracle_200_{}_shot/train_test'.format(self.data_source, self.fd_shot)
            self.all_folders = [os.path.join(self.data_folder, folder) for folder in os.listdir(self.data_folder)]
            self.data_num = 8000
        else:
            raise ValueError('Unrecognized data source')

        random.seed(1)
        random.shuffle(self.all_folders)
        # print(os.getcwd())
        self.tasks_file = '{}W_{}S_{}Q_{}_{}.pkl'.format(self.num_classes, self.k_shot, self.k_query, self.data_source, self.use_set)
        if self.data_source == 'origin':
            self.tasks_file = '{}W_{}S_{}Q_{}_{}.pkl'.format(self.num_classes, self.fd_shot, self.k_query,
                                                             self.data_source, self.use_set)
        print(self.tasks_file)

    def make_tasks(self):
        random.seed(1)
        folders = self.all_folders
        frange = range(len(folders))

        tasks = []
        for _ in tqdm(range(self.data_num)):
            labels_and_images = get_files(folders, frange, self.num_classes, shot=self.k_shot, query=self.k_query,
                                              fd_shot=self.fd_shot)
            labels, rots, images = [x for x in zip(*labels_and_images)]
            labels = np.array(labels, np.long).reshape(self.num_classes, self.num_samples_per_class)
            # print(labels)
            # input('l')
            labels_shot = labels[:, :self.k_shot].reshape(-1)
            labels_query = labels[:, self.k_shot:].reshape(-1)
            rots = np.array(rots, np.long).reshape(self.num_classes, self.num_samples_per_class)
            rots_shot = rots[:, :self.k_shot].reshape(-1)
            rots_query = rots[:, self.k_shot:].reshape(-1)
            images = np.array(images, np.long).reshape(self.num_classes, self.num_samples_per_class)
            images_shot = images[:, :self.k_shot].reshape(-1)
            images_query = images[:, self.k_shot:].reshape(-1)

            shot_idx = np.arange(self.num_shot_per_task)
            np.random.shuffle(shot_idx)
            query_idx = np.arange(self.num_query_per_task)
            np.random.shuffle(query_idx)
            tasks.append((labels_shot[shot_idx], rots_shot[shot_idx], images_shot[shot_idx],
                          labels_query[query_idx], rots_query[query_idx], images_query[query_idx]))

        folders = tuple([x[(len(self.data_folder) + 1):] + '/' + os.listdir(x)[0][-11:-6] for x in folders])
        tasks = tuple(tasks)
        with open(self.tasks_file, 'wb') as f:
            # print(self.tasks_file, 'make_tasks')
            # print(len(self.images))
            pickle.dump((tasks, folders), f)

    def __len__(self):
        if self.use_set == 'train':
            return self.data_num * self.meta_batch_size + 1
        elif self.config['train']:
            self.resume_index = self.resume_val_idx
            return self.data_num * self.meta_batch_size + 1
        else:
            return self.data_num

    def make_one_task(self):
        folders = self.all_folders
        sampled_character_folders = random.sample(folders, self.num_classes)
        random.shuffle(sampled_character_folders)
        labels_and_images = get_images(sampled_character_folders, self.num_classes,
                                       nb_samples=self.num_samples_per_class, shuffle=False)
        label, images = [x for x in zip(*labels_and_images)]
        label = np.array(label)
        images = 1. - (np.concatenate(images, axis=0) / 255.)
        shot_idx = np.arange(self.num_shot_per_task)
        np.random.shuffle(shot_idx)
        query_idx = np.arange(self.num_query_per_task)
        np.random.shuffle(query_idx)
        return torch.FloatTensor(images[:self.k_shot][shot_idx]), torch.LongTensor(label[:self.k_shot][shot_idx]), \
               torch.FloatTensor(images[self.k_shot:][query_idx]), torch.LongTensor(label[self.k_shot:][query_idx])

    def get_tasks(self):
        with open(self.tasks_file, 'rb') as f:
            # print(self.tasks_file)

            self.tasks, self.folders = pickle.load(f)

        index = []
        # print(self.img_num)
        for i in range(self.img_num):
            index.append('0' + str(i) + '.jpg' if i < 10 else str(i) + '.jpg')
        # print(index)
        # print('task-------', self.tasks[0][0])
        # input(1)
        for folder in self.folders:
            folder = folder[:folder.rfind('_') + 1]
            for i in index:
                img_path = os.path.join(self.data_folder, folder + i)
                # print(img_path)
                self.images.append(1 - (np.array(Image.open(img_path))[:, :, 0][np.newaxis, :] / 255.))
        self.images = np.concatenate(self.images, axis=0)
        print('images', len(self.images))

    def __getitem__(self, index):
        index = (index + self.resume_index) % self.data_num
        task = self.tasks[index]

        labels_shot = task[0].astype(np.long)
        # print(labels_shot)
        rots_shot = task[1].astype(np.int)
        index_shot = task[2].astype(np.int)
        images_shot = np.zeros((self.num_shot_per_task, self.img_size[0], self.img_size[1]))
        # print(self.num_shot_per_task)
        # print(index_shot)
        # if self.use_set == 'test':
        #     print(index_shot)
        #     print(self.use_set, len(self.images))
        for i in range(self.num_shot_per_task):
            images_shot[i] = np.rot90(self.images[index_shot[i]], rots_shot[i])
        images_shot = images_shot.reshape(self.num_shot_per_task, 1, self.img_size[0], self.img_size[1])

        labels_query = task[3].astype(np.long)
        rots_query = task[4].astype(np.int)
        index_query = task[5].astype(np.int)
        images_query = np.zeros((self.num_query_per_task, self.img_size[0], self.img_size[1]))
        for i in range(self.num_query_per_task):
            images_query[i] = np.rot90(self.images[index_query[i]], rots_query[i])
        images_query = images_query.reshape(self.num_query_per_task, 1, self.img_size[0], self.img_size[1])

        return (torch.FloatTensor(images_shot), torch.LongTensor(labels_shot),
                torch.FloatTensor(images_query), torch.LongTensor(labels_query))

