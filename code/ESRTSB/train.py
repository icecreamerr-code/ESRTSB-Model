import tensorflow as tf

import sys
import os
sys.path.append('..')
sys.path.append('../..')
from Esrtsb import *
from dataloader import *
import utils as u
from configure.configure import data_set, model_type
import numpy as np
import random
import time
from sklearn.metrics import *
import math
import pickle as pkl
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

random.seed(2022)
np.random.seed(2022)
tf.compat.v1.set_random_seed(2022)

TRAIN_NEG_SAMPLE_NUM = 9
TEST_NEG_SAMPLE_NUM = 99
class FLAGS:
    embedding_dim = 16
    train_batch_size = 100
    valid_batch_size = 100
    model_type = model_type
    data_set = data_set

MAX_LEN = 50
k_hop = 2


## PHP
DATA_DIR_PHP = '../../ESRTSB-data/PHP/feateng/'
GRAPH_DIR_PHP = '../../ESRTSB-data/PHP/feateng/graph/'
USER_PER_COLLECTION_PHP = 500
ITEM_PER_COLLECTION_PHP = 500
START_TIME_PHP = 0
# MAX_1HOP_PHP = 20
# MAX_2HOP_PHP = 20
OBJ_PER_TIME_SLICE_PHP = 10
USER_NUM_PHP = 67093
ITEM_NUM_PHP = 56199
START_TIME_IDX_PHP = 0
TIME_SLICE_NUM_PHP = 27
# TIME_DELTA_PHP = 14
# TIME_SLICE_NUM_PHP = 50
FEAT_SIZE_PHP = 1 + USER_NUM_PHP + ITEM_NUM_PHP
TIME_DELTA_PHP = 14 * 24 * 3600


def restore(model_type, target_file_test, pred_time_test,
            feature_size, eb_dim, max_time_len, lr, reg_lambda, graph_path,
            user_feat, item_feat, user_fnum, item_fnum, time_inter):
    print('restore begin')
    if 'PP' in model_type:
        model = PP(feature_size, eb_dim, max_time_len, user_fnum, item_fnum, MAX_LEN, time_inter, k_hop)
    else:
        print('WRONG MODEL TYPE')
        exit(1)
    model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)

    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        model.restore(sess, 'save_model_{}/{}/ckpt'.format(data_set, model_name))
        print('restore eval begin')
        data_loader_test = Dataloader(FLAGS.valid_batch_size, target_file_test, TEST_NEG_SAMPLE_NUM, max_time_len,
                                      graph_path, user_feat, item_feat, MAX_LEN, pred_time_test, k_hop)
        auc, ndcg_20, ndcg_10,  recall_20, recall_10, mrr, loss = eval(model,
                                                                  sess,
                                                                  data_loader_test,
                                                                  pred_time_test,
                                                                  TEST_NEG_SAMPLE_NUM,
                                                                  reg_lambda,
                                                                  True)
        # p = 1. / (1 + TEST_NEG_SAMPLE_NUM)
        # rig = 1 -(logloss / -(p * math.log(p) + (1 - p) * math.log(1 - p)))
        print(
            'RESTORE, LOSS TEST: %.4f  NDCG@20 TEST: %.4f  NDCG@10 TEST: %.4f  Recall@20 TEST: %.4f  Recall@10 TEST: %.4f  MRR TEST: %.4f AUC TEST: %.4f' % (
                loss, ndcg_20, ndcg_10,  recall_20, recall_10, mrr, auc))
    return model_name, ndcg_20, ndcg_10, recall_20, recall_10, mrr, auc


def getNDCG_at_K(ranklist, target_item, k):
    for i in range(k):
        if ranklist[i] == target_item:
            return math.log(2) / math.log(i + 2)
    return 0


def getRecall_at_K(ranklist, target_item, k):
    if target_item in ranklist[:k]:
        return 1
    else:
        return 0


def getMRR(ranklist, target_item):
    for i in range(len(ranklist)):
        if ranklist[i] == target_item:
            return 1. / (i + 1)
    return 0


def get_ranking_quality(preds, target_iids):
    preds = np.array(preds).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    target_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1).tolist()
    pos_iids = np.array(target_iids).reshape(-1, TEST_NEG_SAMPLE_NUM + 1)[:, 0].flatten().tolist()
    ndcg_20_val = []
    ndcg_10_val = []
    recall_20_val = []
    recall_10_val = []
    mrr_val = []

    for i in range(len(preds)):
        ranklist = list(reversed(np.take(target_iids[i], np.argsort(preds[i]))))
        target_item = pos_iids[i]
        ndcg_20_val.append(getNDCG_at_K(ranklist, target_item, 20))
        ndcg_10_val.append(getNDCG_at_K(ranklist, target_item, 10))
        recall_20_val.append(getRecall_at_K(ranklist, target_item, 20))
        recall_10_val.append(getRecall_at_K(ranklist, target_item, 10))
        mrr_val.append(getMRR(ranklist, target_item))
    return np.mean(ndcg_20_val), np.mean(ndcg_10_val), np.mean(recall_20_val), np.mean(recall_10_val), np.mean(
        mrr_val)


def eval(model, sess, data_loader, pred_time, neg_sample_num, reg_lambda, isTest):
    preds = []
    labels = []
    target_iids = []
    losses = []
    t = time.time()
    a_s = []
    for batch in tqdm(range(data_loader.num_of_batch)):
        uids, iids, uid_seqs, iid_seqs, uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len, label, inter_time,uid_neighbors_seq,uid_neighbors_seq_mask = data_loader.gen_seqs(
            batch)
        feed_dict = u.construct_dict(model, adj_list, user_ids_list, item_ids_list, None, reg_lambda, uids, iids,
                                     uid_seqs, iid_seqs,
                                     label, [pred_time] * len(label), [-1, 1 + neg_sample_num], 0.0, 0.0, 0.0, False,
                                     uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len,
                                     time_list, inter_time,uid_neighbors_seq,uid_neighbors_seq_mask)
        pred, loss = model.eval(sess, feed_dict)
        preds += list(pred)
        labels += label
        losses.append(loss)
        target_iids += iids
    # logloss = log_loss(labels, preds)
    auc = roc_auc_score(labels, preds)
    loss = sum(losses) / len(losses)

    ndcg_20, ndcg_10, recall_20, recall_10, mrr = get_ranking_quality(preds, target_iids)
    print("EVAL TIME: %.4fs" % (time.time() - t))
    return auc, ndcg_20, ndcg_10, recall_20, recall_10, mrr, loss


def train(model_type, feature_size, eb_dim, max_time_len,
          adj_list, user_ids_list, item_ids_list, lr, reg_lambda, target_file_train,
          target_file_valid, pred_time_train, pred_time_valid, graph_path,
          user_feat, item_feat, user_fnum, item_fnum, time_inter):
    if 'PP' in model_type:
        model = PP(feature_size, eb_dim, max_time_len, user_fnum, item_fnum, MAX_LEN, time_inter, k_hop)
    else:
        print('WRONG MODEL TYPE')
        exit(1)

    # gpu settings
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    # training process
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.local_variables_initializer())

        train_losses_step = []
        train_losses = []

        vali_ndcgs_20 = []
        vali_ndcgs_10 = []
        vali_recalls_20 = []
        vali_recalls_10 = []
        vali_mrrs = []
        vali_losses = []
        vali_aucs = []
        step = 0

        data_loader_valid = Dataloader(FLAGS.valid_batch_size, target_file_valid, TEST_NEG_SAMPLE_NUM, max_time_len,
                                       graph_path, user_feat, item_feat, MAX_LEN, pred_time_valid, k_hop)
        vali_auc, vali_ndcg_20, vali_ndcg_10, vali_recall_20, vali_recall_10, vali_mrr, vali_loss = eval(model,
                                                                                                          sess,
                                                                                                          data_loader_valid,
                                                                                                          pred_time_valid,
                                                                                                          TEST_NEG_SAMPLE_NUM,
                                                                                                          reg_lambda,
                                                                                                          False)

        vali_ndcgs_20.append(vali_ndcg_20)
        vali_ndcgs_10.append(vali_ndcg_10)
        vali_recalls_20.append(vali_recall_20)
        vali_recalls_10.append(vali_recall_10)
        vali_mrrs.append(vali_mrr)
        vali_losses.append(vali_loss)
        vali_aucs.append(vali_auc)

        print(
            "STEP %d  LOSS TRAIN: NULL  LOSS VALI: %.4f  NDCG@20 VALI: %.4f  NDCG@10 VALI: %.4f  Recall@20 VALI: %.4f  Recall@10 VALI: %.4f  MRR VALI: %.4f AUC VALI: %.4f" % (
                step, vali_loss, vali_ndcg_20, vali_ndcg_10, vali_recall_20, vali_recall_10, vali_mrr, vali_auc))
        early_stop = False
        data_loader_train = Dataloader(FLAGS.train_batch_size, target_file_train, TRAIN_NEG_SAMPLE_NUM, max_time_len,
                                       graph_path, user_feat, item_feat,
                                       MAX_LEN, pred_time_train, k_hop)
        eval_iter_num = data_loader_train.num_of_batch // 3

        for epoch in tqdm(range(10)):
            if early_stop:
                break
            for batch in range(data_loader_train.num_of_batch):
                if early_stop:
                    break
                target_uids, target_iids, uid_seqs, iid_seqs, uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len, label, inter_time,uid_neighbors_seq,uid_neighbors_seq_mask= data_loader_train.gen_seqs(
                    batch)
                feed_dict_train = u.construct_dict(model, adj_list, user_ids_list, item_ids_list, lr, reg_lambda,
                                                   target_uids, target_iids, uid_seqs, iid_seqs,
                                                   label, [pred_time_train] * len(label),
                                                   [-1, 1 + TRAIN_NEG_SAMPLE_NUM], 0.2, 0.2, 0.2, True,
                                                   uid_which_slices, iid_which_slices, uid_seqs_len, iid_seqs_len,
                                                   time_list, inter_time,uid_neighbors_seq,uid_neighbors_seq_mask)
                train_loss = model.train(sess, feed_dict_train)

                step += 1
                train_losses_step.append(train_loss)
                if step % eval_iter_num == 0:
                    train_loss = sum(train_losses_step) / len(train_losses_step)
                    train_losses.append(train_loss)
                    train_losses_step = []
                    vali_auc, vali_ndcg_20, vali_ndcg_10,  vali_recall_20, vali_recall_10, vali_mrr, vali_loss = eval(
                        model,
                        sess,
                        data_loader_valid,
                        pred_time_valid,
                        TEST_NEG_SAMPLE_NUM,
                        reg_lambda,
                        False)
                    vali_ndcgs_20.append(vali_ndcg_20)
                    vali_ndcgs_10.append(vali_ndcg_10)
                    vali_recalls_20.append(vali_recall_20)
                    vali_recalls_10.append(vali_recall_10)
                    vali_mrrs.append(vali_mrr)
                    vali_losses.append(vali_loss)
                    vali_aucs.append(vali_auc)

                    print(
                        "STEP %d  LOSS TRAIN: %.4f  LOSS VALI: %.4f  NDCG@20 VALI: %.4f  NDCG@10 VALI: %.4f   Recall@20 VALI: %.4f  Recall@10 VALI: %.4f  MRR VALI: %.4f AUC VALI: %.4f" % (
                            step, train_loss, vali_loss, vali_ndcg_20, vali_ndcg_10,  vali_recall_20, vali_recall_10,
                            vali_mrr, vali_auc))

                    if vali_mrrs[-1] > max(vali_mrrs[:-1]):
                        # save model
                        model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)
                        if not os.path.exists('save_model_{}/{}/'.format(data_set, model_name)):
                            os.makedirs('save_model_{}/{}/'.format(data_set, model_name))
                        save_path = 'save_model_{}/{}/ckpt'.format(data_set, model_name)
                        model.save(sess, save_path)

                    if len(vali_mrrs) > 3 and epoch > 0:
                        if (vali_mrrs[-1] < vali_mrrs[-2] and vali_mrrs[-2] < vali_mrrs[-3] and vali_mrrs[-3] <
                                vali_mrrs[-4]):
                            early_stop = True
                            print('=====early stop=====')
                        elif (vali_mrrs[-1] - vali_mrrs[-2]) <= 0.001 and (vali_mrrs[-2] - vali_mrrs[-3]) <= 0.001 and (
                                vali_mrrs[-3] - vali_mrrs[-4]) <= 0.001:
                            early_stop = True
                            print('=====early stop=====')
        # generate log
        if not os.path.exists('logs_{}/'.format(data_set)):
            os.makedirs('logs_{}/'.format(data_set))
        model_name = '{}_{}_{}_{}'.format(model_type, FLAGS.train_batch_size, lr, reg_lambda)

        with open('logs_{}/{}.pkl'.format(data_set, model_name), 'wb') as f:
            dump_tuple = (
                train_losses, vali_losses, vali_ndcgs_20, vali_ndcgs_10, vali_recalls_20, vali_recalls_10,
                vali_mrrs)
            pkl.dump(dump_tuple, f)
        with open('logs_{}/{}.result'.format(data_set, model_name), 'w') as f:
            index = np.argmax(vali_mrrs)
            f.write('Result Validation NDCG@20: {}\n'.format(vali_ndcgs_20[index]))
            f.write('Result Validation NDCG@10: {}\n'.format(vali_ndcgs_10[index]))
            f.write('Result Validation Recall@20: {}\n'.format(vali_recalls_20[index]))
            f.write('Result Validation Recall@10: {}\n'.format(vali_recalls_10[index]))
            f.write('Result Validation MRR: {}\n'.format(vali_mrrs[index]))
        return vali_mrrs[index]


if __name__ == '__main__':

    if data_set == 'PHP':
        user_feat_dict_file = None
        item_feat_dict_file = None
        user_fnum = 1
        item_fnum = 1

        target_file_train = DATA_DIR_PHP + 'target_23.txt'
        target_file_validation = DATA_DIR_PHP + 'target_24_sample.txt'
        target_file_test = DATA_DIR_PHP + 'target_25_sample.txt'

        start_time = START_TIME_PHP
        pred_time_train = 23
        pred_time_validation = 24
        pred_time_test = 25

        # model parameter
        feature_size = FEAT_SIZE_PHP
        max_time_len = TIME_SLICE_NUM_PHP - START_TIME_PHP - 1
        graph_dir = GRAPH_DIR_PHP
        user_num = USER_NUM_PHP
        time_inter = TIME_DELTA_PHP


    else:
        print('WRONG DATASET NAME: {}'.format(data_set))
        exit()

    ################################## training hyper params ##################################
    # TRAINING PROCESS
    lr = 5e-4
    reg_lambdas = [1e-4]
    adj_list = []
    time_list = []
    weight_list = []
    user_ids_list = []
    item_ids_list = []
    size_list = []
    for i in range(max_time_len):
        adj_list.append(u.preprocess_adj(u.load_pickle(graph_dir + 'adj_{}'.format(i))))
        time_list.append(u.load_pickle(graph_dir + 'time_arr_{}'.format(i)))
        user_ids_list.append(u.load_pickle(graph_dir + 'user_ids_{}'.format(i)))
        item_ids_list.append(u.load_pickle(graph_dir + 'item_ids_{}'.format(i)))

    user_feat = None
    item_feat = None
    if user_feat_dict_file:
        user_feat = u.load_pickle(user_feat_dict_file)
    if item_feat_dict_file:
        item_feat = u.load_pickle(item_feat_dict_file)

    print(f'dataset:{data_set},model:{model_type}')
    ndcg_20_list = []
    ndcg_10_list = []
    recall_20_list = []
    recall_10_list = []
    mrr_list = []
    for num in range(3):
        vali_mrrs = []
        hyper_list = []
        for reg_lambda in reg_lambdas:
            vali_mrr = train(model_type, feature_size, FLAGS.embedding_dim, max_time_len, adj_list,
                             user_ids_list, item_ids_list, lr, reg_lambda, target_file_train, target_file_validation,
                             pred_time_train, pred_time_validation, graph_dir,
                             user_feat, item_feat, user_fnum, item_fnum, time_inter)
            vali_mrrs.append(vali_mrr)
            hyper_list.append(reg_lambda)

        index = np.argmax(vali_mrrs)
        best_hyper = hyper_list[index]
        reg_lambda = 1e-4
        model_name, ndcg_20, ndcg_10, recall_20, recall_10, mrr, auc = restore(model_type, target_file_test, pred_time_test,
                                                                           feature_size, FLAGS.embedding_dim,
                                                                           max_time_len,
                                                                           lr, reg_lambda, graph_dir, user_feat,
                                                                           item_feat, user_fnum, item_fnum, time_inter)

        ndcg_20_list.append(ndcg_20)
        ndcg_10_list.append(ndcg_10)
        recall_20_list.append(recall_20)
        recall_10_list.append(recall_10)
        mrr_list.append(mrr)

    with open('logs_{}/{}.test.result'.format(data_set, model_name), 'w') as f:
        f.write('Result Test NDCG@20: {}\n'.format(np.mean(ndcg_20_list)))
        f.write('Result Test NDCG@10: {}\n'.format(np.mean(ndcg_10_list)))
        f.write('Result Test Recall@20: {}\n'.format(np.mean(recall_20_list)))
        f.write('Result Test Recall@10: {}\n'.format(np.mean(recall_10_list)))
        f.write('Result Test MRR: {}\n'.format(np.mean(mrr_list)))

    print('Result Test NDCG@20: {}\n'.format(np.mean(ndcg_20_list)))
    print('Result Test NDCG@10: {}\n'.format(np.mean(ndcg_10_list)))
    print('Result Test Recall@20: {}\n'.format(np.mean(recall_20_list)))
    print('Result Test Recall@10: {}\n'.format(np.mean(recall_10_list)))
    print('Result Test MRR: {}\n'.format(np.mean(mrr_list)))




