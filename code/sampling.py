import random
import sys
import os
from configure.configure import data_set

# PHP dataset parameters
DATA_DIR_PHP = '../ESRTSB-data/PHP/feateng/'

random.seed(2022)


def sample_files(target_file,
                 user_seq_file,
                 item_seq_file,
                 sample_target_file,
                 sample_user_seq_file,
                 sample_item_seq_file,
                 sample_factor):
    '''
     #随机采样（1/sample factor）*taget_num个样本用于验证和测试
    :param target_file:
    :param user_seq_file:
    :param item_seq_file:
    :param sample_target_file:
    :param sample_user_seq_file:
    :param sample_item_seq_file:
    :param sample_factor:
    :return:
    '''
    print('sampling begin')
    target_lines = open(target_file).readlines()
    # user_seq_lines = open(user_seq_file).readlines()
    # item_seq_lines = open(item_seq_file).readlines()

    sample_target_lines = []
    # sample_user_seq_lines = []
    # sample_item_seq_lines = []

    length = len(target_lines)
    for i in range(length):
        rand_int = random.randint(1, sample_factor)
        if rand_int == 1:
            sample_target_lines.append(target_lines[i])
            # sample_user_seq_lines.append(user_seq_lines[i])
            # sample_item_seq_lines.append(item_seq_lines[i])

    with open(sample_target_file, 'w') as f:
        f.writelines(sample_target_lines)
    # with open(sample_user_seq_file, 'w') as f:
    #     f.writelines(sample_user_seq_lines)
    # with open(sample_item_seq_file, 'w') as f:
    #     f.writelines(sample_item_seq_lines)
    print('sampling end')


if __name__ == "__main__":

    dataset = data_set

    if dataset == 'PHP':
        # PHP
        sample_files(DATA_DIR_PHP + 'target_50.txt',
                     DATA_DIR_PHP + 'validation_user_hist_seq_50.txt',
                     DATA_DIR_PHP + 'validation_item_hist_seq_50.txt',
                     DATA_DIR_PHP + 'target_50_sample.txt',
                     DATA_DIR_PHP + 'validation_user_hist_seq_50_sample.txt',
                     DATA_DIR_PHP + 'validation_item_hist_seq_50_sample.txt',
                     60)
        sample_files(DATA_DIR_PHP + 'target_51.txt',
                     DATA_DIR_PHP + 'test_user_hist_seq_51.txt',
                     DATA_DIR_PHP + 'test_item_hist_seq_51.txt',
                     DATA_DIR_PHP + 'target_51_sample.txt',
                     DATA_DIR_PHP + 'test_user_hist_seq_51_sample.txt',
                     DATA_DIR_PHP + 'test_item_hist_seq_51_sample.txt',
                     60)

    else:
        print('WRONG DATASET: {}'.format(dataset))

