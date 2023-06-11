import random
# import pymongo
import pickle as pkl
import time
import numpy as np
import multiprocessing
import datetime
import sys
sys.path.append('..')
import utils as u
from configure.configure import data_set
random.seed(11)

NEG_SAMPLE_NUM = 99
SECONDS_PER_DAY = 24*3600


DATA_DIR_PHP = '../ESRTSB-data/PHP/feateng/'
USER_PER_COLLECTION_PHP = 500
ITEM_PER_COLLECTION_PHP = 500
START_TIME_PHP = 1400668490.0
OBJ_PER_TIME_SLICE_PHP = 10
USER_NUM_PHP = 6781
ITEM_NUM_PHP = 1159
START_TIME_IDX_PHP = 0
TIME_SLICE_NUM_PHP = 53
TIME_DELTA_PHP = 7


DATA_DIR_JavaScript = '../ESRTSB-data/JavaScript/feateng/'
USER_PER_COLLECTION_JavaScript = 500
ITEM_PER_COLLECTION_JavaScript = 500
START_TIME_JavaScript = 1400668490.0
OBJ_PER_TIME_SLICE_JavaScript = 10
USER_NUM_JavaScript= 12436
ITEM_NUM_JavaScript = 2639
START_TIME_IDX_JavaScript = 0
TIME_SLICE_NUM_JavaScript = 27
TIME_DELTA_JavaScript = 14

DATA_DIR_Ruby = '../ESRTSB-data/Ruby/feateng/'
USER_PER_COLLECTION_Ruby = 500
ITEM_PER_COLLECTION_Ruby = 500
START_TIME_Ruby = 1400668490.0
OBJ_PER_TIME_SLICE_Ruby = 10
USER_NUM_Ruby= 6945
ITEM_NUM_Ruby = 1724
START_TIME_IDX_Ruby = 0
TIME_SLICE_NUM_Ruby = 53
TIME_DELTA_Ruby = 7

class TargetGen(object):
    def __init__(self, user_neg_dict_file, path_to_store, user_num, item_num, user_per_collection,
                item_per_collection, start_time, start_time_idx, time_delta):
        if user_neg_dict_file != None:
            with open(user_neg_dict_file, 'rb') as f:
                self.user_neg_dict = pkl.load(f)  
        else:
            self.user_neg_dict = {}

        # url = "mongodb://localhost:27017/"
        # client = pymongo.MongoClient(url)
        # db = client[db_name]
        self.user_num = user_num
        self.item_num = item_num
        
        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection

        user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            user_coll_num += 1
        item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            item_coll_num += 1

        # self.user_colls = [db['user_%d'%(i)] for i in range(user_coll_num)]
        # self.item_colls = [db['item_%d'%(i)] for i in range(item_coll_num)]
        self.user_colls = [u.load_pickle(path_to_store + 'user_seq_%d' % i) for i in range(user_coll_num)]
        self.item_colls = [u.load_pickle(path_to_store + 'item_seq_%d' % i) for i in range(item_coll_num)]

        self.start_time = start_time
        self.start_time_idx = start_time_idx
        self.time_delta = time_delta

    def gen_user_neg_items(self, uid, neg_sample_num, start_iid, end_iid, pop_items):
        '''
        user的item负采样，通过随机选取user-item一跳外的99个item，作为负样本
        :param uid:
        :param neg_sample_num:
        :param start_iid:
        :param end_iid:
        :param pop_items:
        :return:
        '''
        if str(uid) in self.user_neg_dict:
            user_neg_list = self.user_neg_dict[str(uid)]
        else:
            user_neg_list = []
        
        if len(user_neg_list) >= neg_sample_num:
            return user_neg_list[:neg_sample_num]
        else:
            if pop_items == None:
                for i in range(neg_sample_num - len(user_neg_list)):
                    user_neg_list.append(str(random.randint(start_iid, end_iid)))
                return user_neg_list
            else:
                pop_items_len = len(pop_items)
                for i in range(neg_sample_num - len(user_neg_list)):
                    user_neg_list.append(pop_items[random.randint(0, pop_items_len-1)])
                return user_neg_list

    def gen_target_file(self, neg_sample_num, target_file, user_hist_dict_file, pred_time, pop_items_file = None):
        if pop_items_file != None:
            with open(pop_items_file, 'rb') as f:
                pop_items = pkl.load(f)
        else:
            pop_items = None
        
        with open(user_hist_dict_file, 'rb') as f:
            user_hist_dict = pkl.load(f)

        target_lines = []
        for user_coll in self.user_colls:
            # cursor = user_coll.find({})
            for user_doc in user_coll:
                if user_doc['1hop'][pred_time] != []:
                    uid = user_doc['uid']
                    if str(uid) in user_hist_dict:
                        pos_iids = user_doc['1hop'][pred_time]
                        pos_iid = pos_iids[0][0]

                        neg_iids = self.gen_user_neg_items(uid, neg_sample_num, self.user_num + 1, self.user_num + self.item_num, pop_items)
                        # target_lines.append(','.join([str(uid),str(pos_iid)] + neg_iids) + '\n')
                        target_lines.append(','.join([str(uid), str(pred_time),str(pos_iid)] + neg_iids) + '\n')
        with open(target_file, 'w') as f:
            f.writelines(target_lines)
        print('generate {} completed'.format(target_file))

    def gen_user_item_hist_dict_ccmr(self, hist_file, user_hist_dict_file, item_hist_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}

        # load and construct dicts
        with open(hist_file, 'r') as f:
            for line in f:
                uid, iid, _, time_str = line[:-1].split(',')
                uid = str(int(uid) + 1)
                iid = str(int(iid) + 1 + self.user_num)
                time_int = int(time.mktime(datetime.datetime.strptime(time_str, "%Y-%m-%d").timetuple()))
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if time_idx < self.start_time_idx:
                    continue
                if time_idx >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, time_int)]
                else:
                    user_hist_dict[uid].append((iid, time_int))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, time_int)]
                else:
                    item_hist_dict[iid].append((uid, time_int))
            print('dicts construct completed')
        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)

    def gen_user_item_hist_dict_taobao(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            f.readline()
            for line in f:
                uid, iid, _, timestamp_str = line[:-1].split('\t')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                timestamp = int(float(timestamp_str))
                time_idx = int((timestamp - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, timestamp)]
                else:
                    user_hist_dict[uid].append((iid, timestamp))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, timestamp)]
                else:
                    item_hist_dict[iid].append((uid, timestamp))
            print('dicts construct completed')

        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)

    def gen_user_item_hist_dict_tmall(self, hist_file, user_hist_dict_file, item_hist_dict_file, remap_dict_file, pred_time):
        user_hist_dict = {}
        item_hist_dict = {}
        
        with open(remap_dict_file, 'rb') as f:
            uid_remap_dict = pkl.load(f)
            iid_remap_dict = pkl.load(f)

        # load and construct dicts
        with open(hist_file, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                uid, iid, rid, time_stamp = line[:-1].split('\t')
                uid = uid_remap_dict[uid]
                iid = iid_remap_dict[iid]

                time_int = float(time_stamp)
                time_idx = int((time_int - self.start_time) / (SECONDS_PER_DAY * self.time_delta))
                if int(time_idx) < self.start_time_idx:
                    continue
                if int(time_idx) >= pred_time:
                    continue
                if uid not in user_hist_dict:
                    user_hist_dict[uid] = [(iid, time_int)]
                else:
                    user_hist_dict[uid].append((iid, time_int))
                if iid not in item_hist_dict:
                    item_hist_dict[iid] = [(uid, time_int)]
                else:
                    item_hist_dict[iid].append((uid, time_int))
            print('dicts construct completed')

        # sort by time
        for uid in user_hist_dict.keys():
            user_hist_dict[uid] = sorted(user_hist_dict[uid], key=lambda tup:tup[1])
        for iid in item_hist_dict.keys():
            item_hist_dict[iid] = sorted(item_hist_dict[iid], key=lambda tup:tup[1])
        print('sort completed')

        # new dict
        user_hist_dict_sort = {}
        item_hist_dict_sort = {}
        for uid in user_hist_dict.keys():
            user_hist_dict_sort[uid] = [tup[0] for tup in user_hist_dict[uid]]
        for iid in item_hist_dict.keys():
            item_hist_dict_sort[iid] = [tup[0] for tup in item_hist_dict[iid]]
        print('new dict completed')

        # dump
        with open(user_hist_dict_file, 'wb') as f:
            pkl.dump(user_hist_dict_sort, f)
        with open(item_hist_dict_file, 'wb') as f:
            pkl.dump(item_hist_dict_sort, f)
    
    def gen_pop_items(self, pop_items_file, pop_standard, max_iid):#timeslice 有6个以上有行为记录
        pop_items = []
        for item_coll in self.item_colls:
            # cursor = item_coll.find({})
            for item_doc in item_coll:
                num_not_null_slice = 0
                for nei in item_doc['1hop']:
                    if nei != []:
                        num_not_null_slice += 1
                if num_not_null_slice >= pop_standard and item_doc['iid'] <= max_iid:
                    pop_items.append(str(item_doc['iid']))
        print('num of pop_items: {}'.format(len(pop_items)))
        with open(pop_items_file, 'wb') as f:
            pkl.dump(pop_items, f)


if __name__ == '__main__':

    dataset= data_set
    if dataset == 'PHP':
        # PHP
        tg = TargetGen(None, DATA_DIR_PHP+'1hop/', user_num = USER_NUM_PHP,
                    item_num = ITEM_NUM_PHP, user_per_collection = USER_PER_COLLECTION_PHP,
                    item_per_collection = ITEM_PER_COLLECTION_PHP, start_time = START_TIME_PHP,
                    start_time_idx = START_TIME_IDX_PHP, time_delta = TIME_DELTA_PHP)
        tg.gen_pop_items(DATA_DIR_PHP + 'pop_items.pkl', 5, 1 + USER_NUM_PHP + ITEM_NUM_PHP)
        #user_hist_dict_9.pkl：user: history behavior sequence
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_PHP + 'filtered_user_behavior.txt', DATA_DIR_PHP + 'user_hist_dict_51.pkl', DATA_DIR_PHP + 'item_hist_dict_51.pkl', DATA_DIR_PHP + 'remap_dict.pkl', 51)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_PHP + 'filtered_user_behavior.txt', DATA_DIR_PHP + 'user_hist_dict_50.pkl', DATA_DIR_PHP + 'item_hist_dict_50.pkl', DATA_DIR_PHP + 'remap_dict.pkl', 50)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_PHP + 'filtered_user_behavior.txt', DATA_DIR_PHP + 'user_hist_dict_49.pkl', DATA_DIR_PHP + 'item_hist_dict_49.pkl', DATA_DIR_PHP + 'remap_dict.pkl', 49)

        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_PHP + 'target_51.txt', DATA_DIR_PHP + 'user_hist_dict_51.pkl', 51, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_PHP + 'target_50.txt', DATA_DIR_PHP + 'user_hist_dict_50.pkl', 50, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_PHP + 'target_49.txt', DATA_DIR_PHP + 'user_hist_dict_49.pkl', 49, None)
    elif data_set == 'JavaScript':
        tg = TargetGen(None, DATA_DIR_JavaScript + '1hop/', user_num=USER_NUM_JavaScript,
                       item_num=ITEM_NUM_JavaScript, user_per_collection=USER_PER_COLLECTION_JavaScript,
                       item_per_collection=ITEM_PER_COLLECTION_JavaScript, start_time=START_TIME_JavaScript,
                       start_time_idx=START_TIME_IDX_JavaScript, time_delta=TIME_DELTA_JavaScript)
        tg.gen_pop_items(DATA_DIR_JavaScript + 'pop_items.pkl', 5, 1 + USER_NUM_JavaScript + ITEM_NUM_JavaScript)
        # user_hist_dict_9.pkl：user: history behavior sequence
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_JavaScript + 'filtered_user_behavior.txt',
                                          DATA_DIR_JavaScript + 'user_hist_dict_25.pkl',
                                          DATA_DIR_JavaScript + 'item_hist_dict_25.pkl',
                                          DATA_DIR_JavaScript + 'remap_dict.pkl', 25)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_JavaScript + 'filtered_user_behavior.txt',
                                          DATA_DIR_JavaScript + 'user_hist_dict_24.pkl',
                                          DATA_DIR_JavaScript + 'item_hist_dict_24.pkl',
                                          DATA_DIR_JavaScript + 'remap_dict.pkl', 24)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_JavaScript + 'filtered_user_behavior.txt',
                                          DATA_DIR_JavaScript + 'user_hist_dict_23.pkl',
                                          DATA_DIR_JavaScript + 'item_hist_dict_23.pkl',
                                          DATA_DIR_JavaScript + 'remap_dict.pkl', 23)

        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_JavaScript + 'target_25.txt',
                           DATA_DIR_JavaScript + 'user_hist_dict_25.pkl', 25, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_JavaScript + 'target_24.txt',
                           DATA_DIR_JavaScript + 'user_hist_dict_24.pkl', 24, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_JavaScript + 'target_23.txt',
                           DATA_DIR_JavaScript + 'user_hist_dict_23.pkl', 23, None)
    elif data_set == 'Ruby':
        tg = TargetGen(None, DATA_DIR_Ruby + '1hop/', user_num=USER_NUM_Ruby,
                       item_num=ITEM_NUM_Ruby, user_per_collection=USER_PER_COLLECTION_Ruby,
                       item_per_collection=ITEM_PER_COLLECTION_Ruby, start_time=START_TIME_Ruby,
                       start_time_idx=START_TIME_IDX_Ruby, time_delta=TIME_DELTA_Ruby)
        tg.gen_pop_items(DATA_DIR_Ruby + 'pop_items.pkl', 5, 1 + USER_NUM_Ruby + ITEM_NUM_Ruby)
        # user_hist_dict_9.pkl：user: history behavior sequence
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Ruby + 'filtered_user_behavior.txt',
                                          DATA_DIR_Ruby + 'user_hist_dict_51.pkl',
                                          DATA_DIR_Ruby + 'item_hist_dict_51.pkl', DATA_DIR_Ruby + 'remap_dict.pkl',
                                          51)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Ruby + 'filtered_user_behavior.txt',
                                          DATA_DIR_Ruby + 'user_hist_dict_50.pkl',
                                          DATA_DIR_Ruby + 'item_hist_dict_50.pkl', DATA_DIR_Ruby + 'remap_dict.pkl',
                                          50)
        tg.gen_user_item_hist_dict_taobao(DATA_DIR_Ruby + 'filtered_user_behavior.txt',
                                          DATA_DIR_Ruby + 'user_hist_dict_49.pkl',
                                          DATA_DIR_Ruby + 'item_hist_dict_49.pkl', DATA_DIR_Ruby + 'remap_dict.pkl',
                                          49)

        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Ruby + 'target_51.txt', DATA_DIR_Ruby + 'user_hist_dict_51.pkl',
                           51, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Ruby + 'target_50.txt', DATA_DIR_Ruby + 'user_hist_dict_50.pkl',
                           50, None)
        tg.gen_target_file(NEG_SAMPLE_NUM, DATA_DIR_Ruby + 'target_49.txt', DATA_DIR_Ruby + 'user_hist_dict_49.pkl',
                           49, None)
        #target_11.txt: uid,pos_iid,neg_iid
    else:
        print('WRONG DATASET: {}'.format(dataset))

