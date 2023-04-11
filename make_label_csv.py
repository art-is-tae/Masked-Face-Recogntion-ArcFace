import pandas as pd
import os

def make_train_csv(train_root_path = './train/', save_path = './label/', name = 'CASIA_masked'):
    file_idx = []
    file_targets = []
    file_names = []
    file_path = []

    idx = -1
    target = -1

    for subdir in os.listdir(train_root_path):
        pathes = os.path.join(train_root_path, subdir)
        # target
        target += 1
        for filename in os.listdir(pathes):
            # id
            idx += 1
            file_names.append(filename)
            file_targets.append(target)
            file_idx.append(idx)
            file_path.append(pathes + '/' + filename)

    # make dataframe

    data_dic = {
        'target': file_targets,
        'filename': file_names,
        'path': file_path
    }

    df = pd.DataFrame(data_dic)

    print(df)
    # save CSV
    df.to_csv(save_path + 'train_' + name + '.csv')
    return df


def make_test_csv(test_root_path = './test/', txt_path = './label/masked_pairs_modified.txt', save_path = './label/', name = 'RWMFVD'):
    # Test (Real-world masked face verification dataset)

    # create target dictionary
    class_dict = {}
    idx = 0
    for subdir in os.listdir(test_root_path):
        class_dict[subdir] = idx
        idx += 1

    # dataframe list
    pair1_list = []
    pair2_list = []
    ans_list = []

    # read_text_file
    f = open(txt_path, 'r')
    while True:
        # load text file
        line = f.readline()
        line_list = []
        if not line: break
        contents = line.split()
        for content in contents:
            classes = content.split('/')
            line_list.append(classes)
        pair1 = line_list[0]
        pair2 = line_list[1]
        ans = line_list[2]
        
        pair1_list.append(pair1)
        pair2_list.append(pair2)
        ans_list.append(ans)
    f.close()

    # target encoding
    target_list = []

    for i in range(len(pair1_list)):
        cls = pair1_list[i][0]
        cls_id = class_dict[cls]
        target_list.append(cls_id)
        
    pair_target_list = []

    for i in range(len(pair2_list)):
        cls = pair2_list[i][0]
        cls_id = class_dict[cls]
        pair_target_list.append(cls_id)

    # target path encoding
    path_list = []
    for i in range(len(pair1_list)):
        target_root_path = './test/'
        target_path = target_root_path + pair1_list[i][0] + '/' + pair1_list[i][1]
        path_list.append(target_path)

    pair_path_list = []
    for i in range(len(pair2_list)):
        target_root_path = './test/'
        target_path = target_root_path + pair2_list[i][0] + '/' + pair2_list[i][1]
        pair_path_list.append(target_path)

    answer_list = []

    for i in range(len(ans_list)):
        answer_list.append(int(*ans_list[i]))
    
    # make dataframe
    data_dic = {
        'target': target_list,
        'path': path_list,
        'pair_target': pair_target_list,
        'pair_path': pair_path_list,
        'answer': answer_list,
    }

    df = pd.DataFrame(data_dic)

    # save CSV
    df.to_csv(save_path + 'test_' + name + '.csv')
    return df

def make_eval_csv(test_root_path = './test/', txt_path = './label/masked_pairs_modified.txt', save_path = './label/', name = 'RWMFVD'):
    # Test (Real-world masked face verification dataset)

    # create target dictionary
    class_dict = {}
    idx = 0
    for subdir in os.listdir(test_root_path):
        class_dict[subdir] = idx
        idx += 1

    # dataframe list
    pair1_list = []
    pair2_list = []
    ans_list = []

    # read_text_file
    f = open(txt_path, 'r')
    while True:
        # load text file
        line = f.readline()
        line_list = []
        if not line: break
        contents = line.split()
        for content in contents:
            classes = content.split('/')
            line_list.append(classes)
        pair1 = line_list[0]
        pair2 = line_list[1]
        ans = line_list[2]
        
        pair1_list.append(pair1)
        pair2_list.append(pair2)
        ans_list.append(ans)
    f.close()

    # target encoding
    target_list = []
    for i in range(len(pair1_list)):
        cls = pair1_list[i][0]
        cls_id = class_dict[cls]
        target_list.append(cls_id)
        
    pair_target_list = []
    for i in range(len(pair2_list)):
        cls = pair2_list[i][0]
        cls_id = class_dict[cls]
        pair_target_list.append(cls_id)

    # target path encoding
    path_list = []
    for i in range(len(pair1_list)):
        target_root_path = './test/'
        target_path = target_root_path + pair1_list[i][0] + '/' + pair1_list[i][1]
        path_list.append(target_path)

    pair_path_list = []
    for i in range(len(pair2_list)):
        target_root_path = './test/'
        target_path = target_root_path + pair2_list[i][0] + '/' + pair2_list[i][1]
        pair_path_list.append(target_path)

    answer_list = []

    for i in range(len(ans_list)):
        answer_list.append(int(*ans_list[i]))

    diff_target_list = []
    diff_path_list = []
    diff_pair_target_list = []
    diff_pair_path_list = []
    same_target_list = []
    same_path_list = []
    same_pair_target_list = []
    same_pair_path_list = []
    for i in range(len(answer_list)):
        if answer_list[i] == 1: # same
            same_target_list.append(target_list[i])
            same_path_list.append(path_list[i])
            same_pair_target_list.append(pair_target_list[i])
            same_pair_path_list.append(pair_path_list[i])
        else: # diff
            diff_target_list.append(target_list[i])
            diff_path_list.append(path_list[i])
            diff_pair_target_list.append(pair_target_list[i])
            diff_pair_path_list.append(pair_path_list[i])
    
    # make dataframe
    data_dic_same = {
        'target': same_target_list,
        'path': same_path_list,
        'pair_target': same_pair_target_list,
        'pair_path': same_pair_path_list,
    }
    
    data_dic_diff = {
        'target': diff_target_list,
        'path': diff_path_list,
        'pair_target': diff_pair_target_list,
        'pair_path': diff_pair_path_list,
    }

    df_same = pd.DataFrame(data_dic_same)
    df_diff = pd.DataFrame(data_dic_diff)

    # save CSV
    df_same.to_csv(save_path + 'eval_same_' + name + '.csv')
    df_diff.to_csv(save_path + 'eval_diff_' + name + '.csv')

    return df_same, df_diff

if __name__ == "__main__":
    make_test_csv()
    make_train_csv()
    make_eval_csv()