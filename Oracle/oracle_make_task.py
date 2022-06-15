import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from DataSolver import *
import random
import numpy as np

# 5-way 1-shot 1-query


def test_way_shot_query_orginaldata(fd_shot=3, way=20, shot=1, query=1, data_num=10000, source='origin'):
    print('1111')
    random.seed(2022)
    np.random.seed(2022)
    # print(os.getcwd())
    img_num = fd_shot
    dataset = Oracle(use_set='train', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': data_num, 'train': True, 'img_num': img_num, 'data_source': source})
    dataset.make_tasks()

    # random.seed(2)
    # np.random.seed(2)
    # dataset.change_use_set('val')
    # dataset.make_tasks()

    random.seed(2022)
    np.random.seed(2022)
    dataset = Oracle(use_set='test', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': 8000, 'train': False, 'data_source': source})
    dataset.make_tasks()


def test_way_shot_query_entiredata(way=20, shot=1, query=1, data_num=20000, source='entire'):
    print('2222')
    random.seed(2022)
    np.random.seed(2022)
    # print(os.getcwd())

    img_num = 21 * shot
    fd_shot = shot
    dataset = Oracle(use_set='train', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': data_num, 'train': True, 'img_num': img_num, 'data_source': source})
    dataset.make_tasks()

    # random.seed(2)
    # np.random.seed(2)
    # dataset.change_use_set('val')
    # dataset.make_tasks()

    random.seed(2022)
    np.random.seed(2022)
    dataset = Oracle(use_set='test', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': 8000, 'train': False, 'data_source': source})
    dataset.make_tasks()




def test_way_shot_query_npzdata(way=20, shot=1, query=1, data_num=20000, source='npz'):
    print('3333')
    random.seed(2022)
    np.random.seed(2022)
    # print(os.getcwd())

    img_num = 41 * shot
    fd_shot = shot
    dataset = Oracle(use_set='train', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': data_num, 'train': True, 'img_num': img_num, 'data_source': source})
    dataset.make_tasks()

    # random.seed(2)
    # np.random.seed(2)
    # dataset.change_use_set('val')
    # dataset.make_tasks()

    random.seed(2022)
    np.random.seed(2022)
    dataset = Oracle(use_set='test', config={'num_classes': way, 'k_shot': shot, 'k_query': query, 'fd_shot': fd_shot, 'data_num': 8000, 'train': False, 'data_source': source})
    dataset.make_tasks()



if __name__ == '__main__':
    test_way_shot_query_orginaldata(fd_shot=3, way=20, shot=1, query=1, data_num=10000, source='origin')
    test_way_shot_query_orginaldata(fd_shot=5, way=20, shot=1, query=1, data_num=10000, source='origin')
    test_way_shot_query_entiredata(way=20, shot=1, query=1, data_num=20000, source='entire')
    test_way_shot_query_entiredata(way=20, shot=3, query=3, data_num=20000, source='entire')
    test_way_shot_query_entiredata(way=20, shot=5, query=5, data_num=20000, source='entire')
    test_way_shot_query_npzdata(way=200, shot=1, query=1, data_num=40000, source='npz')
    test_way_shot_query_npzdata(way=200, shot=3, query=3, data_num=40000, source='npz')
    test_way_shot_query_npzdata(way=200, shot=5, query=5, data_num=40000, source='npz')

