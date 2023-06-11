import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import datetime
import time
import matplotlib.pyplot as plt
import pandas as pd
SOCIAL_NETWORK_FILE="../ESRTSB-data/followers.csv"
RAW_DIR = '../ESRTSB-data/Ruby/'
FEATENG_DIR = '../ESRTSB-data/Ruby/feateng/'

START_TIME = 1400668490#1245568000.0
END_TIME = 1432204490#1277568000.0
TIME_DELTA = 7 * 24 * 3600

def join_user_profile(user_profile_file, behavior_file, joined_file):
    user_profile_dict = {}
    with open(user_profile_file, 'r') as f:
        for line in f:
            uid, aid, gid = line[:-1].split(',')
            user_profile_dict[uid] = ','.join([aid, gid])
    
    # join
    newlines = []
    with open(behavior_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            user_profile = user_profile_dict[uid]
            newlines.append(line[:-1] + ',' + user_profile + '\n')
    with open(joined_file, 'w') as f:
        f.writelines(newlines)


def static_data(raw_file):
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        x=[]
        for line in lines:
            uid, iid, rid, time_stamp = line[:-1].split('\t')
            x.append(float(time_stamp))
        print(min(x))
        print(max(x))
def reset_id(data, id_map, column_name='UserId'):
    mapped_id = data[column_name].map(id_map)
    data[column_name] = mapped_id
    return data
def process_social(raw_file,out_file): # read in social network.
    df_net = pd.read_csv(SOCIAL_NETWORK_FILE, sep=',', dtype={0:str, 1: str})
    df_net.columns=["Follower","Followee","Created_at"]
    df_net.drop_duplicates(subset=['Follower', 'Followee'], inplace=True)
    print(df_net.shape)
    friend_size = df_net.groupby('Follower').size()
    #net = net[np.in1d(net.Follower, friend_size[friend_size>=5].index)]
    print(df_net.head(5))
    print('Statistics of social network:')
    print('\tTotal user in social network:{}.\n\tTotal edges(links) in social network:{}.'.format( \
        df_net.Follower.nunique(), len(df_net)))
    print('\tAverage number of friends for users: {}'.format(df_net.groupby('Follower').size().mean()))
    df=pd.read_csv(raw_file,sep=',', dtype={0:str, 1: str,2:str,3:str,4:str})
    df.columns=["ItemId","repo_name","language","UserId","TimeStamp"]
    df['TimeStamp']=df['TimeStamp'].apply(lambda x:time.mktime(time.strptime(x.replace('"',""),"%Y-%m-%d %H:%M:%S")))
    df = df[df['TimeStamp'].between(START_TIME, END_TIME, inclusive=True)]
    print(df.head(5))
    print('Statistics of user ratings:')
    print('\tNumber of total ratings: {}'.format(len(df)))
    print('\tNumber of users: {}'.format(df.UserId.nunique()))
    print('\tNumber of items: {}'.format(df.ItemId.nunique()))
    df_net = df_net.loc[df_net['Follower'].isin(df['UserId'].unique())]  # 得到有session记录的user
    # print(df_net.loc[df_net['Follower']=="1619308"])

    df_net = df_net.loc[df_net['Followee'].isin(df['UserId'].unique())]  # 得到有session记录的user friends
    print(df_net.shape)
    df=df.loc[df['UserId'].isin(df_net.Follower.unique())]
    print(df.shape)
    df["Rate"] = 1
    df=df[["UserId","ItemId","Rate","TimeStamp"]]

    df.to_csv(out_file,sep=',',index=False)
    df_net=df_net[["Follower","Followee"]]
    df_net.to_csv(RAW_DIR+'followers.csv',sep=',',index=False)
    return df,df_net




def preprocess_raw_data(raw_file, out_file, remap_dict_file, plt_file, user_feat_dict_file, item_feat_dict_file):
    time_idxs = []
    uid_set = set()
    iid_set = set()
    rid_set = set()
    sid_set = set()
    bid_set = set()
    aid_set = set()
    gid_set = set()
    newlines_filtered=[]
    with open(raw_file, 'r',encoding="utf-8") as f:
        f.readline()
        lines = f.readlines()

        for line in lines:
            uid, iid,rid, time_stamp = line.strip().split(',')

            # time_stamp=time.mktime(time.strptime(time_stamp,'%Y-%m-%d %H:%M:%S'))
            # if float(time_stamp) < START_TIME or float(time_stamp) > END_TIME:
            #     continue
            uid_set.add(uid)
            iid_set.add(iid)
            rid_set.add(rid)
            newlines_filtered.append('\t'.join([uid, iid, rid, time_stamp]) + '\n')
            date_str = float(time_stamp)
            # time_int = int(time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple()))
            time_int = date_str
            t_idx = (time_int - START_TIME) // TIME_DELTA
            time_idxs.append(int(t_idx))

    with open(FEATENG_DIR + 'filtered_user_behavior.txt', 'w') as f:
        f.write('\t'.join(['UserId', 'ItemId', 'Rate', 'Timestamp']) + '\n')
        f.writelines(newlines_filtered)
    # remap
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    rid_list = list(rid_set)
    # sid_list = list(sid_set)
    # bid_list = list(bid_set)
    # aid_list = list(aid_set)
    # gid_list = list(gid_set)

    print('user num: {}'.format(len(uid_list)))
    print('item num: {}'.format(len(iid_list)))
    print('rate num: {}'.format(len(rid_list)))

    
    remap_id = 1
    uid_remap_dict = {}
    iid_remap_dict = {}
    rid_remap_dict = {}


    for uid in uid_list:
        uid_remap_dict[uid] = str(remap_id)
        remap_id += 1
    df_net = pd.read_csv(RAW_DIR + 'followers.csv', sep=',', dtype={0: str, 1: str})
    df_net = df_net.loc[df_net['Follower'].isin(uid_list)]
    df_net = df_net.loc[df_net['Followee'].isin(uid_list)]
    print(df_net.shape)
    df_net=reset_id(df_net,uid_remap_dict,column_name='Follower')
    df_net=reset_id(df_net,uid_remap_dict,column_name='Followee')
    df_net.to_csv(FEATENG_DIR+'social_net.csv',index=False)
    for iid in iid_list:
        iid_remap_dict[iid] = str(remap_id)
        remap_id += 1
    for rid in rid_list:
        rid_remap_dict[rid] = str(remap_id)
        remap_id += 1

    print('feat size: {}'.format(remap_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(rid_remap_dict, f)

    print('remap ids completed')

    # remap file generate
    item_feat_dict = {}
    user_feat_dict = {}
    # for dummy user
    user_feat_dict['0'] = [0, 0]
    newlines = []
    # with open(raw_file, 'r') as f:
    with open(FEATENG_DIR + 'filtered_user_behavior.txt', 'r') as f:

        lines = f.readlines()[1:]
        for i in range(len(lines)):
            uid, iid, rid, time_stamp = lines[i][:-1].split('\t')
            uid_remap = uid_remap_dict[uid]
            iid_remap = iid_remap_dict[iid]
            rid_remap = rid_remap_dict[rid]

            t_idx = time_idxs[i]
            item_feat_dict[iid_remap] = [int(rid_remap)]
            # user_feat_dict[uid_remap] = [int(aid_remap), int(gid_remap)]
            newlines.append(','.join([uid_remap, iid_remap, '_', str(t_idx)]) + '\n')
    with open(out_file, 'w') as f:
        f.writelines(newlines)
    print('remaped file generated')


    with open(user_feat_dict_file, 'wb') as f:
        pkl.dump(user_feat_dict, f)
    print('user feat dict dump completed')
    with open(item_feat_dict_file, 'wb') as f:
        pkl.dump(item_feat_dict, f)
    print('item feat dict dump completed')

    # plot distribution
    print('max t_idx: {}'.format(max(time_idxs)))
    print('min t_idx: {}'.format(min(time_idxs)))



if __name__ == "__main__":

    process_social(RAW_DIR + 'Ruby.csv',RAW_DIR+'Ruby_filterd.csv')
    preprocess_raw_data(RAW_DIR + 'Ruby_filterd.csv', FEATENG_DIR + 'remaped_user_behavior.csv', FEATENG_DIR + 'remap_dict.pkl', FEATENG_DIR + 'time_distri.png', FEATENG_DIR + 'user_feat_dict.pkl', FEATENG_DIR + 'item_feat_dict.pkl')


