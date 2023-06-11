import random
import pickle as pkl
import time
import numpy as np
import multiprocessing
import sys

sys.path.append('..')
from configure.configure import data_set
import utils as u
import scipy.sparse as sp
import pandas as pd
WORKER_N = 5




DATA_DIR_PHP = '../ESRTSB-data/PHP/feateng/'
USER_PER_COLLECTION_PHP = 500
ITEM_PER_COLLECTION_PHP = 500
MAX_1HOP_PHP = 20
MAX_2HOP_PHP = 20
OBJ_PER_TIME_SLICE_PHP = 10
USER_NUM_PHP = 6781
ITEM_NUM_PHP = 1159
START_TIME_IDX_PHP = 0
TIME_SLICE_NUM_PHP = 53
TIME_DELTA_PHP = 7
START_TIME_PHP=0

DATA_DIR_JavaScript = '../ESRTSB-data/JavaScript/feateng/'
USER_PER_COLLECTION_JavaScript = 500
ITEM_PER_COLLECTION_JavaScript = 500
MAX_1HOP_JavaScript = 20
MAX_2HOP_JavaScript = 20
OBJ_PER_TIME_SLICE_JavaScript = 10
USER_NUM_JavaScript= 12436
ITEM_NUM_JavaScript = 2639
START_TIME_IDX_JavaScript = 0
TIME_SLICE_NUM_JavaScript = 27
TIME_DELTA_JavaScript = 14
START_TIME_JavaScript=0


DATA_DIR_Ruby = '../ESRTSB-data/Ruby/feateng/'
USER_PER_COLLECTION_Ruby = 500
ITEM_PER_COLLECTION_Ruby = 500
MAX_1HOP_Ruby = 20
MAX_2HOP_Ruby = 20
OBJ_PER_TIME_SLICE_Ruby = 10
USER_NUM_Ruby= 6945
ITEM_NUM_Ruby = 1724
START_TIME_IDX_Ruby = 0
TIME_SLICE_NUM_Ruby = 53
TIME_DELTA_Ruby = 7
START_TIME_Ruby=0

class GraphHandler(object):
    def __init__(self,
                 time_slice_num,
                 path_to_store,
                 user_num,
                 item_num,
                 start_time,
                 user_per_collection,
                 item_per_collection,
                 path_to_social_net
                 ):
        self.user_num = user_num
        self.item_num = item_num
        self.start_time = start_time
        self.time_slice_num = time_slice_num

        self.user_per_collection = user_per_collection
        self.item_per_collection = item_per_collection

        self.user_coll_num = self.user_num // self.user_per_collection
        if self.user_num % self.user_per_collection != 0:
            self.user_coll_num += 1
        self.item_coll_num = self.item_num // self.item_per_collection
        if self.item_num % self.item_per_collection != 0:
            self.item_coll_num += 1
        self.user_colls = [u.load_pickle(path_to_store + 'user_seq_%d' % i) for i in range(self.user_coll_num)]
        self.item_colls = [u.load_pickle(path_to_store + 'item_seq_%d' % i) for i in range(self.item_coll_num)]
        self.path_to_social_net=path_to_social_net
    def reset_id(self,data, id_map, column_name='UserId'):
        mapped_id = data[column_name].map(id_map)
        data[column_name] = mapped_id
        return data
    def build_graph(self, graph_path, user_feat, item_feat):
        for i in range(self.time_slice_num - 1):

            slice_set_user = []
            slice_set_item = []
            for item_coll in self.item_colls:

                for item in item_coll:
                    # print(item)
                    if len(item['1hop'][i]) > 0:
                        slice_set_item.append(item['iid'])
                        slice_set_user.extend([j[0] for j in item['1hop'][i]])

            for user_coll in self.user_colls:
                for user in user_coll:
                    if len(user['1hop'][i]) > 0:
                        slice_set_user.append(user['uid'])
                        slice_set_item.extend([j[0] for j in user['1hop'][i]])

            uid_length = len(set(slice_set_user))
            iid_length = len(set(slice_set_item))
            remap_id = 0
            rows_user = []
            rows_item = []
            uid_remap_dict = {}
            iid_remap_dict = {}
            slice_set_user = list(set(slice_set_user))
            slice_set_item = list(set(slice_set_item))
            for uid in slice_set_user:
                uid_remap_dict[uid] = remap_id
                rows_user.append(uid)
                remap_id += 1
            social_net = pd.read_csv(self.path_to_social_net)
            social_net = social_net.loc[social_net['Follower'].isin(slice_set_user)]
            social_net = social_net.loc[social_net['Followee'].isin(slice_set_user)]
            print(social_net.shape)
            # social_net=self.reset_id(social_net,uid_remap_dict,column_name="Follower")
            user_neighbors=u.UniformNeighborSampler(social_net,uid_remap_dict).construct_adj()
            for iid in slice_set_item:
                iid_remap_dict[iid] = remap_id
                rows_item.append(iid)
                remap_id += 1

            row = []
            col = []
            weight = []
            time_array = np.zeros((uid_length + iid_length, 1))
            for user_coll in self.user_colls:
                for user in user_coll:
                    if len(user['1hop'][i]) > 0:
                        hop_list = user['1hop'][i]
                        for iid in hop_list:
                            col.append(iid_remap_dict[iid[0]])
                        length = len(hop_list)
                        row.extend([uid_remap_dict[user['uid']]] * length)
                        time_array[uid_remap_dict[user['uid']], 0] = max([j[1] for j in hop_list])
                        weight.extend([1] * length)

            for item_coll in self.item_colls:
                for item in item_coll:
                    if len(item['1hop'][i]) > 0:
                        hop_list = item['1hop'][i]
                        for uid in hop_list:
                            col.append(uid_remap_dict[uid[0]])
                        length = len(hop_list)
                        row.extend([iid_remap_dict[item['iid']]] * length)
                        time_array[iid_remap_dict[item['iid']], 0] = max([j[1] for j in hop_list])
                        weight.extend([1] * length)

            with open(graph_path + 'remap_dict_file_uid_{}'.format(i), 'wb') as f:
                pkl.dump(uid_remap_dict, f)
            with open(graph_path + 'remap_dict_file_iid_{}'.format(i), 'wb') as f:
                pkl.dump(iid_remap_dict, f)
            with open(graph_path + 'user_neighbors_{}'.format(i), 'wb') as f:
                pkl.dump(user_neighbors, f)
            print('remap ids completed')

            adj = sp.csr_matrix(
                (weight, (row, col)), shape=(uid_length + iid_length, uid_length + iid_length))


            u.dump_pickle(graph_path + 'adj_{}'.format(i), adj)
            u.dump_pickle(graph_path + 'time_arr_{}'.format(i), time_array)

            print('adj matrix completed')

            print(len(rows_user))
            print(len(rows_item))

            target_user_batch = []
            target_item_batch = []
            if user_feat == None:
                for uid in rows_user:
                    target_user_batch.append([uid])
            else:
                for uid in rows_user:
                    target_user_batch.append([uid] + user_feat[str(uid)])
            if item_feat == None:
                for iid in rows_item:
                    target_item_batch.append([iid])
            else:
                for iid in rows_item:
                    target_item_batch.append([iid] + item_feat[str(iid)])

            u.dump_pickle(graph_path + 'user_ids_{}'.format(i), target_user_batch)
            u.dump_pickle(graph_path + 'item_ids_{}'.format(i), target_item_batch)
            # break


if __name__ == "__main__":

    data_set = data_set
    if data_set == 'PHP':
        path_to_store = '../ESRTSB-data/PHP/feateng/1hop/'
        path_to_social_net = '../ESRTSB-data/PHP/feateng/social_net.csv'
        graph_path = '../ESRTSB-data/PHP/feateng/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_PHP, path_to_store, USER_NUM_PHP,
                                     ITEM_NUM_PHP,
                                     START_TIME_PHP, USER_PER_COLLECTION_PHP, ITEM_PER_COLLECTION_PHP,path_to_social_net)

    elif data_set == 'JavaScript':
        path_to_store = '../ESRTSB-data/JavaScript/feateng/1hop/'
        path_to_social_net = '../ESRTSB-data/JavaScript/feateng/social_net.csv'
        graph_path = '../ESRTSB-data/JavaScript/feateng/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_JavaScript, path_to_store, USER_NUM_JavaScript,
                                     ITEM_NUM_JavaScript,
                                     START_TIME_JavaScript, USER_PER_COLLECTION_JavaScript, ITEM_PER_COLLECTION_JavaScript,path_to_social_net)


    elif data_set == 'Ruby':
        path_to_store = '../ESRTSB-data/Ruby/feateng/1hop/'
        path_to_social_net = '../ESRTSB-data/Ruby/feateng/social_net.csv'
        graph_path = '../ESRTSB-data/Ruby/feateng/graph/'
        user_feat_dict_file = None
        item_feat_dict_file = None
        graph_handler = GraphHandler(TIME_SLICE_NUM_Ruby, path_to_store, USER_NUM_Ruby,
                                     ITEM_NUM_Ruby,
                                     START_TIME_Ruby, USER_PER_COLLECTION_Ruby,
                                     ITEM_PER_COLLECTION_Ruby, path_to_social_net)

    user_feat = None
    item_feat = None
    if not user_feat_dict_file == None:
        user_feat = u.load_pickle(user_feat_dict_file)
    if not item_feat_dict_file == None:
        item_feat = u.load_pickle(item_feat_dict_file)

    graph_handler.build_graph(graph_path, user_feat, item_feat)

