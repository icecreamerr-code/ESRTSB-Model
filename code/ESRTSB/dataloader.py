import sys
sys.path.append('../')
import utils as u
import random
import numpy as np
random.seed(2022)

class Dataloader(object):
    def __init__(self,batch_size,target_file,neg_sample_num,max_time,graph_path,user_feat,item_feat,max_len,pred_time,k_hop,max_degree=20):
        self.batch_size = batch_size
        self.neg_sample_num = neg_sample_num
        self.max_time = max_time
        self.remap_file_uids=[]
        self.remap_file_iids=[]
        self.remap_uid_neighbors=[]
        self.pred_time = pred_time
        self.graph_path = graph_path
        self.k_hop = k_hop
        self.max_degree=max_degree

        for i in range(self.max_time):
            self.remap_file_uids.append(u.load_pickle(graph_path+'remap_dict_file_uid_{}'.format(i)))
        for i in range(self.max_time):
            self.remap_file_iids.append(u.load_pickle(graph_path+'remap_dict_file_iid_{}'.format(i)))
        for i in range(self.max_time):
            self.remap_uid_neighbors.append(u.load_pickle(graph_path+'user_neighbors_{}'.format(i)))

        if self.batch_size % (1+self.neg_sample_num) != 0:
            print('batch size should be time of {}'.format(1 + self.neg_sample_num))
            exit(1)
        self.batch_size2line_num = int(self.batch_size / (1 + self.neg_sample_num))
        with open(target_file, 'r') as f:
            self.target_lines = f.readlines()
        self.num_of_batch = len(self.target_lines) // self.batch_size2line_num
        print(self.num_of_batch)
        if self.num_of_batch * self.batch_size2line_num < len(self.target_lines):
            self.num_of_batch += 1
        self.user_feat=user_feat
        self.item_feat=item_feat
        self.max_len = max_len


    def producer(self,prod_batch_num):
        uids=[]
        iids=[]
        time=[]
        label=[]


        if (prod_batch_num + 1) * self.batch_size2line_num <= len(self.target_lines):
            lines = self.target_lines[prod_batch_num * self.batch_size2line_num: (prod_batch_num + 1) * self.batch_size2line_num]
        else:
            lines = self.target_lines[prod_batch_num * self.batch_size2line_num:]

        target_user=[]
        src_list = []
        for line in lines:
            line_list = line[:-1].split(',')
            uids += [line_list[0]]*(1 + self.neg_sample_num)
            time += [line_list[1]]*(1 + self.neg_sample_num)
            if not self.user_feat == None:
                target_user.extend([[int(line_list[0])]+self.user_feat[line_list[0]]]*(1 + self.neg_sample_num))
            else:
                target_user.extend([[int(line_list[0])]]*(1 + self.neg_sample_num))
            iids += line_list[2:(3 + self.neg_sample_num)]
            label.append(1)
            label.extend([0]*self.neg_sample_num)


        uids = [int(uid) for uid in uids]
        iids = [int(iid) for iid in iids]

        return uids,iids,label,target_user,time

    def gen_user_seq(self,uids):
        uid_seqs=[]
        uid_neighbors_seq=[]
        uid_neighbors_seq_mask=[]
        uid_which_slices=[]
        uid_seqs_len = []
        num=0

        for uid in uids:
            uid_seq=[]
            uid_neighbor_seq = []
            uid_which_slice = []
            for i in range(self.pred_time):
                if uid in self.remap_file_uids[i].keys():
                    uid_seq.append(self.remap_file_uids[i][uid])

                    uid_neighbor_seq.append(self.remap_uid_neighbors[i][self.remap_file_uids[i][uid]])

                    mask=np.ones(self.max_degree,dtype=np.float32)
                    length = np.count_nonzero(self.remap_uid_neighbors[i][self.remap_file_uids[i][uid]])
                    mask[length:] = 0
                    uid_neighbors_seq_mask.append(mask)
                    uid_which_slice.append(i+num*self.max_time)#uid:[t0,t1,...,tm]
                else:
                    uid_seq.append(0)
                    uid_neighbor_seq.append([0]*self.max_degree)
                    mask = np.zeros(self.max_degree, dtype=np.float32)
                    uid_neighbors_seq_mask.append(mask)
            num+=1
            uid_seqs.append(uid_seq+(self.max_time-len(uid_seq))*[0])
            # print((self.max_time - len(uid_neighbor_seq)) * [[0] * self.max_degree],'bbbb')
            uid_neighbors_seq.append(uid_neighbor_seq+(self.max_time-len(uid_neighbor_seq))*[[0]*self.max_degree])
            uid_seqs_len.append(len(uid_which_slice))
            uid_which_slices.append(uid_which_slice+(self.max_time-len(uid_which_slice))*[0])

        return uid_seqs,uid_which_slices,uid_seqs_len,uid_neighbors_seq,uid_neighbors_seq_mask

    def gen_item_seq(self,iids):
        iid_seqs=[]
        iid_which_slices = []
        iid_seqs_len = []
        num=0
        for iid in iids:
            iid_seq=[]
            iid_which_slice = []
            for i in range(self.pred_time):
                if iid in self.remap_file_iids[i].keys():
                    iid_seq.append(self.remap_file_iids[i][iid])
                    iid_which_slice.append(i+num*self.max_time)
                else:
                    iid_seq.append(0)
            num+=1
            iid_seqs.append(iid_seq+(self.max_time-len(iid_seq))*[0])
            iid_seqs_len.append(len(iid_which_slice))
            iid_which_slices.append(iid_which_slice+(self.max_time-len(iid_which_slice))*[0])

        return iid_seqs,iid_which_slices,iid_seqs_len

    def gen_seqs(self,prod_batch_num):
        uids, iids, label, target_user_batch, time = self.producer(prod_batch_num)
        uid_seqs, uid_which_slices, uid_seqs_len,uid_neighbors_seq,uid_neighbors_seq_mask = self.gen_user_seq(uids)
        iid_seqs, iid_which_slices, iid_seqs_len = self.gen_item_seq(iids)
        target_item_batch=[]
        if self.item_feat == None:
            for iid in iids:
                target_item_batch.append([iid])
        else:
            for iid in iids:
                target_item_batch.append([iid]+self.item_feat[str(iid)])
        # print(np.array(uid_neighbors_seq).shape)
        return target_user_batch,target_item_batch,uid_seqs,iid_seqs,uid_which_slices,iid_which_slices,uid_seqs_len,iid_seqs_len,label,time,uid_neighbors_seq,uid_neighbors_seq_mask

