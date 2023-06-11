# import pymongo
import pickle as pkl
import datetime
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
sys.path.append('../')
import utils as u
from configure.configure import data_set


random.seed(11)

SECONDS_PER_DAY = 24 * 3600



DATA_DIR_PHP = '../ESRTSB-data/PHP/feateng/'
USER_PER_COLLECTION_PHP = 500
ITEM_PER_COLLECTION_PHP = 500
START_TIME_PHP = 0
MAX_1HOP_PHP = 20
MAX_2HOP_PHP = 20
OBJ_PER_TIME_SLICE_PHP = 10
USER_NUM_PHP = 6781
ITEM_NUM_PHP = 1159
START_TIME_IDX_PHP = 0
TIME_SLICE_NUM_PHP = 53
TIME_DELTA_PHP = 7

DATA_DIR_JavaScript = '../ESRTSB-data/JavaScript/feateng/'
USER_PER_COLLECTION_JavaScript = 500
ITEM_PER_COLLECTION_JavaScript = 500
START_TIME_JavaScript = 0
MAX_1HOP_JavaScript = 20
MAX_2HOP_JavaScript = 20
OBJ_PER_TIME_SLICE_JavaScript = 10
USER_NUM_JavaScript = 12436
ITEM_NUM_JavaScript = 2639
START_TIME_IDX_JavaScript = 0
TIME_SLICE_NUM_JavaScript = 27
TIME_DELTA_JavaScript = 14

DATA_DIR_Ruby = '../ESRTSB-data/Ruby/feateng/'
USER_PER_COLLECTION_Ruby = 500
ITEM_PER_COLLECTION_Ruby = 500
START_TIME_Ruby = 0
MAX_1HOP_Ruby = 20
MAX_2HOP_Ruby = 20
OBJ_PER_TIME_SLICE_Ruby = 10
USER_NUM_Ruby = 6945
ITEM_NUM_Ruby = 1724
START_TIME_IDX_Ruby = 0
TIME_SLICE_NUM_Ruby = 53
TIME_DELTA_Ruby = 7


class GraphStore(object):
    def __init__(self, rating_file, user_per_collection = USER_PER_COLLECTION_PHP,
                item_per_collection = ITEM_PER_COLLECTION_PHP,  start_time = START_TIME_PHP,
                max_1hop = MAX_1HOP_PHP, max_2hop = MAX_2HOP_PHP, user_num = USER_NUM_PHP,
                item_num = ITEM_NUM_PHP, db_1hop = '1hop', db_2hop = '2hop',
                time_slice_num = TIME_SLICE_NUM_PHP):

        self.db_1hop = db_1hop

        self.user_num = user_num
        self.item_num = item_num

        # input files
        self.rating_file = open(rating_file, 'r')

        # about time index
        self.time_slice_num = time_slice_num

        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection
        self.start_time = start_time
        self.max_1hop = max_1hop
        self.max_2hop = max_2hop


    def gen_user_doc(self, uid):
        user_doc = {}
        user_doc['uid'] = uid
        user_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        return user_doc

    def gen_item_doc(self, iid):
        item_doc = {}
        item_doc['iid'] = iid
        item_doc['1hop'] = [[] for i in range(self.time_slice_num)]
        return item_doc

    def construct_coll_1hop(self):
        list_of_user_doc_list = []
        list_of_item_doc_list = []

        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1
        # print(user_coll_num)
        for i in range(user_coll_num):
            user_doc_list = []
            for uid in range(i * self.user_per_collection + 1, (i + 1) * self.user_per_collection + 1):
                user_doc_list.append(self.gen_user_doc(uid))
            list_of_user_doc_list.append(user_doc_list)
        # print(list_of_user_doc_list)
        for i in range(item_coll_num):
            item_doc_list = []
            for iid in range(i * self.item_per_collection + 1 + self.user_num, (i + 1) * self.item_per_collection + 1 + self.user_num):
                item_doc_list.append(self.gen_item_doc(iid))
            list_of_item_doc_list.append(item_doc_list)

        for line in self.rating_file:
            #1 hop，因为是用户地直接点击记录
            #分桶：user的不同时间片的所有item:[T1[item1,...],T2[item3,...],...,[T12]]
            #     item:[T1[u1,...],T2[u3,...],...,[u12]]
            uid, iid, _, t_idx = line[:-1].split(',')
            # print('xxxxxxxxxxxxxxx')
            # print(list_of_user_doc_list[(int(uid) - 1) // self.user_per_collection])
            # print(uid,iid,'xxxxxxxxxxxxxx',(int(uid) - 1) // self.user_per_collection,t_idx)
            list_of_user_doc_list[(int(uid) - 1) // self.user_per_collection][(int(uid) - 1) % self.user_per_collection]['1hop'][int(t_idx)].append([int(iid),t_idx])
            list_of_item_doc_list[(int(iid) - self.user_num - 1) // self.item_per_collection][(int(iid) - self.user_num - 1) % self.item_per_collection]['1hop'][int(t_idx)].append([int(uid),t_idx])
        
        print('user and item doc list completed')

        for i in range(len(list_of_user_doc_list)):
            # self.db_1hop['user_%d'%(i)].insert_many(list_of_user_doc_list[i])
            u.dump_pickle(self.db_1hop+'user_seq_%d'%i,list_of_user_doc_list[i])
        print('user collection completed')
        for i in range(len(list_of_item_doc_list)):
            # self.db_1hop['item_%d'%(i)].insert_many(list_of_item_doc_list[i])
            u.dump_pickle(self.db_1hop + 'item_seq_%d' % i, list_of_item_doc_list[i])
        print('item collection completed')
    
    def construct_coll_2hop(self):
        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1
        
        user_colls = [self.db_1hop['user_%d'%i] for i in range(user_coll_num)]
        item_colls = [self.db_1hop['item_%d'%i] for i in range(item_coll_num)]

        all_user_docs = []
        all_item_docs = []
        for user_coll in user_colls:
            cursor = user_coll.find({})
            for user_doc in cursor:
                all_user_docs.append(user_doc)
        for item_coll in item_colls:
            cursor = item_coll.find({})
            for item_doc in cursor:
                all_item_docs.append(item_doc)
        print('loading 1hop graph data completed')
        

        # gen item 2hop
        print('item 2 hop gen begin')
        for i in range(item_coll_num):
            item_docs_block = []
            #过滤一跳的用户，+ self.user_num +
            for iid in range(1 + self.user_num + i * self.item_per_collection, 1 + self.user_num + (i + 1) * self.item_per_collection):
                # 过滤一跳的用户，+ self.user_num +
                old_item_doc = all_item_docs[iid - 1 - self.user_num]
                new_item_doc = {
                    'iid': iid,
                    '1hop': old_item_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(self.start_time):
                    new_item_doc['2hop'].append([])
                    new_item_doc['degrees'].append([])
                for t in range(self.start_time, self.time_slice_num):
                    iids_2hop = []
                    degrees_2hop = []
                    #开始获取一跳的用户的连接，也就是跟一跳用户相连的2跳用户
                    uids = old_item_doc['1hop'][t]
                    if len(uids) > self.max_1hop:
                        random.shuffle(uids)
                        uids = uids[:self.max_1hop]
                    for uid in uids:
                        user_doc = all_user_docs[uid - 1]#获取1跳用户的T1时段的items，再根据该items查找其item1跳的T1时段的user，得到2跳item
                        degree = len(user_doc['1hop'][t])
                        if degree > 1 and degree <= self.max_1hop:
                            iids_2hop += user_doc['1hop'][t]
                            degrees_2hop += [degree] * degree
                        elif degree > self.max_1hop:
                            iids_2hop += user_doc['1hop'][t][:self.max_1hop]
                            degrees_2hop += [degree] * self.max_1hop
                        else:
                            continue

                    if len(iids_2hop) > self.max_2hop:#2跳item数量大于阈值，随机选择阈值数量的item，以及更新相应的degree
                        idx = np.random.choice(np.arange(len(iids_2hop)), len(iids_2hop), replace=False)
                        iids_2hop = np.array(iids_2hop)[idx].tolist()[:self.max_2hop]
                        degrees_2hop = np.array(degrees_2hop)[idx].tolist()[:self.max_2hop]

                    new_item_doc['2hop'].append(iids_2hop)
                    new_item_doc['degrees'].append(degrees_2hop)

                item_docs_block.append(new_item_doc)
            self.db_2hop['item_%d'%i].insert_many(item_docs_block)
            print('item block-{} completed'.format(i))
        print('item 2 hop gen completed')

        # gen user 2hop
        print('user 2 hop gen begin')
        for i in range(user_coll_num):
            user_docs_block = []
            for uid in range(1 + i * self.user_per_collection, 1 + (i + 1) * self.user_per_collection):
                old_user_doc = all_user_docs[uid - 1]
                new_user_doc = {
                    'uid': uid,
                    '1hop': old_user_doc['1hop'],
                    '2hop': [],
                    'degrees': []
                }
                for t in range(self.start_time):
                    new_user_doc['2hop'].append([])
                    new_user_doc['degrees'].append([])
                for t in range(self.start_time, self.time_slice_num):
                    uids_2hop = []
                    degrees_2hop = []

                    iids = old_user_doc['1hop'][t]
                    if len(iids) > self.max_1hop:
                        random.shuffle(iids)
                        iids = iids[:self.max_1hop]
                    for iid in iids:
                        item_doc = all_item_docs[iid - 1 - self.user_num]
                        degree = len(item_doc['1hop'][t])
                        if degree > 1 and degree <= self.max_1hop:
                            uids_2hop += item_doc['1hop'][t]
                            degrees_2hop += [degree] * degree
                        elif degree > self.max_1hop:
                            uids_2hop += item_doc['1hop'][t][:self.max_1hop]
                            degrees_2hop += [degree] * self.max_1hop
                        else:
                            continue
                        
                    if len(uids_2hop) > self.max_2hop:
                        idx = np.random.choice(np.arange(len(uids_2hop)), len(uids_2hop), replace=False)
                        uids_2hop = np.array(uids_2hop)[idx].tolist()[:self.max_2hop]
                        degrees_2hop = np.array(degrees_2hop)[idx].tolist()[:self.max_2hop]

                    new_user_doc['2hop'].append(uids_2hop)
                    new_user_doc['degrees'].append(degrees_2hop)


                user_docs_block.append(new_user_doc)
            self.db_2hop['user_%d'%i].insert_many(user_docs_block)
            print('user block-{} completed'.format(i))
        print('user 2 hop gen completed')


    def cal_stat(self):
        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1
        
        user_colls = [self.db_1hop['user_%d'%i] for i in range(user_coll_num)]
        item_colls = [self.db_1hop['item_%d'%i] for i in range(item_coll_num)]
        
        # calculate user doc
        hist_len_user = []
        for user_coll in user_colls:
            for user_doc in user_coll.find({}):
                for t in range(self.time_slice_num):
                    hist_len_user.append(len(user_doc['1hop'][t]))
        
        arr = np.array(hist_len_user)
        print('max user slice hist len: {}'.format(np.max(arr)))
        print('min user slice hist len: {}'.format(np.min(arr)))
        print('null slice per user: {}'.format(arr[arr == 0].size / self.user_num))
        print('small(<=5) slice per user: {}'.format(arr[arr <= 5].size / self.user_num))
        print('mean user slice(not null) hist len: {}'.format(np.mean(arr[arr > 0])))

        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)

        
        print('-------------------------------------')
        # calculate item doc
        hist_len_item = []
        for item_coll in item_colls:
            for item_doc in item_coll.find({}):
                for t in range(self.time_slice_num):
                    hist_len_item.append(len(item_doc['1hop'][t]))
        arr = np.array(hist_len_item)
        print('max item hist len: {}'.format(np.max(arr)))
        print('min item hist len: {}'.format(np.min(arr)))
        print('null per item: {}'.format(arr[arr == 0].size / self.item_num))
        print('small(<=5) per item: {}'.format(arr[arr <= 5].size / self.item_num))
        print('mean item hist(not null) len: {}'.format(np.mean(arr[arr > 0])))
        
        arr = arr.reshape(-1, self.time_slice_num)
        arr = np.sum(arr, axis=0)
        print(arr)


if __name__ == "__main__":

    dataset = data_set

    if dataset == 'PHP':
        # For PHP
        gs = GraphStore(DATA_DIR_PHP + 'remaped_user_behavior.csv', user_per_collection = USER_PER_COLLECTION_PHP,
                    item_per_collection = ITEM_PER_COLLECTION_PHP,  start_time = START_TIME_PHP,
                    max_1hop = MAX_1HOP_PHP, max_2hop = MAX_2HOP_PHP, user_num = USER_NUM_PHP,
                    item_num = ITEM_NUM_PHP, db_1hop = DATA_DIR_PHP+'1hop/', db_2hop = DATA_DIR_PHP+'2hop/',
                    time_slice_num = TIME_SLICE_NUM_PHP)
        gs.construct_coll_1hop()
    elif dataset == 'JavaScript':
        gs = GraphStore(DATA_DIR_JavaScript + 'remaped_user_behavior.csv', user_per_collection = USER_PER_COLLECTION_JavaScript,
                    item_per_collection = ITEM_PER_COLLECTION_JavaScript,  start_time = START_TIME_JavaScript,
                    max_1hop = MAX_1HOP_JavaScript, max_2hop = MAX_2HOP_JavaScript, user_num = USER_NUM_JavaScript,
                    item_num = ITEM_NUM_JavaScript, db_1hop = DATA_DIR_JavaScript+'1hop/', db_2hop = DATA_DIR_JavaScript+'2hop/',
                    time_slice_num = TIME_SLICE_NUM_JavaScript)
        gs.construct_coll_1hop()
    elif dataset == 'Ruby':
        gs = GraphStore(DATA_DIR_Ruby + 'remaped_user_behavior.csv',
                        user_per_collection=USER_PER_COLLECTION_Ruby,
                        item_per_collection=ITEM_PER_COLLECTION_Ruby, start_time=START_TIME_Ruby,
                        max_1hop=MAX_1HOP_Ruby, max_2hop=MAX_2HOP_Ruby, user_num=USER_NUM_Ruby,
                        item_num=ITEM_NUM_Ruby, db_1hop=DATA_DIR_Ruby + '1hop/',
                        db_2hop=DATA_DIR_Ruby + '2hop/',
                        time_slice_num=TIME_SLICE_NUM_Ruby)
        gs.construct_coll_1hop()
    else:
        print('WRONG DATASET: {}'.format(dataset))