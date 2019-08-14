import numpy as np
import random
import scipy.io as scio
import os


root = './datasets'
assert os.path.isdir(root), "not found datasets in path: {}".format(root)


def transformer_bbcsport_2():
    '''
    对bbcsport 2 view 数据集的loader & 转换
    :param data: data['X'].shape = (dims, simple nums), data['gt'].shape = (1, simple nums)
    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                totalnum = round((raw.shape[0]) / 2)  # train len
    '''

    dataFile = os.path.join(root,'bbcsport_2view.mat')
    data = scio.loadmat(dataFile)
    # 分离X1，X2，并且使其为数组
    X_raw = data['X']
    X1_raw = X_raw[0, 0]
    X2_raw = X_raw[0, 1]
    X1_raw = X1_raw.toarray()
    X2_raw = X2_raw.toarray()
    X1_raw = np.transpose(X1_raw)
    X2_raw = np.transpose(X2_raw)
    # 分离Y
    Y_raw = data['gt']
    # 合并成一个列表并且进行抽样，得出S1（算法训练集）和S2（算法测试集）（数组）
    raw = np.hstack((X1_raw, X2_raw, Y_raw))
    totalnum = round((raw.shape[0]) / 2)  # train len
    X1_num = X1_raw.shape[1]
    X2_num = X2_raw.shape[1]
    raw = raw.tolist()
    S1 = random.sample(raw, totalnum)
    S2 = random.sample(raw, totalnum)
    S1 = np.array(S1)
    S2 = np.array(S2)
    # train set
    S1_X = S1[:, 0:X1_num + X2_num]
    S1_Y = S1[:, X1_num + X2_num:]
    # test set
    S2_X = S2[:, 0:X1_num + X2_num]
    S2_Y = S2[:, X1_num + X2_num:]

    return S1_X,S1_Y,S2_X,S2_Y,totalnum


def transformer_3Source():
    '''
    transformer of dataset: 3Source
    :param  data['BBC']  channel 1, shape=(n_simples, dims)
            data['Guardian'] channel 2, shape=(n_simples, dims)
            data[Reuters']
            data['y'] label, shape=(n_simples, 1)
    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                totalnum
    '''
    dataFile = os.path.join(root,'3Sources.mat')
    data = scio.loadmat(dataFile)

    X_1 = data['BBC']
    X_2 = data['Guardian']
    X_3 = data['Reuters']
    label = data['y']
    n_simples = data['y'].shape[0]

    X1_dims = X_1.shape[1]
    X2_dims = X_2.shape[1]
    X3_dims = X_3.shape[1]

    raw = np.hstack((X_1, X_2, X_3,label))

    totalnum = round((raw.shape[0]) / 2)  # train len

    raw = raw.tolist()
    train_set = np.array(random.sample(raw, totalnum))
    test_set = np.array(random.sample(raw, totalnum))

    S1_X = train_set[:, :X1_dims+X2_dims+X3_dims]
    S1_Y = train_set[:, X1_dims+X2_dims+X3_dims].reshape(-1,1)
    S2_X = test_set[:, :X1_dims+X2_dims+X3_dims]
    S2_Y = test_set[:, X1_dims+X2_dims+X3_dims].reshape(-1,1)

    return S1_X,S1_Y,S2_X,S2_Y,totalnum


def transformer_Caltech101_7():
    '''
    transformer of dataset: Caltech101_7
    :param  data['gist']  channel 1, shape=(n_simples, dims)
            data['hog'] channel 2, shape=(n_simples, dims)
            data['lbp']
            data['y'] label, shape=(n_simples, 1)

            ! classes = 7 (Y.max()==7)

    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                train_nums  训练集数量
    '''
    dataFile = os.path.join(root, 'Caltech101-7.mat')
    data = scio.loadmat(dataFile)

    X_1 = data['gist']
    X_2 = data['hog']
    X_3 = data['lbp']
    label = data['y']
    n_simples = data['y'].shape[0]

    X1_dims = X_1.shape[1]
    X2_dims = X_2.shape[1]
    X3_dims = X_3.shape[1]

    raw = np.hstack((X_1, X_2, X_3, label))
    train_nums = round((raw.shape[0]) / 2)  # train len
    raw = raw.tolist()
    train_set = np.array(random.sample(raw, train_nums))
    test_set = np.array(random.sample(raw, train_nums))

    S1_X = train_set[:, :X1_dims + X2_dims + X3_dims]
    S1_Y = train_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)
    S2_X = test_set[:, :X1_dims + X2_dims + X3_dims]
    S2_Y = test_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)

    return S1_X, S1_Y, S2_X, S2_Y, train_nums


def transformer_Caltech101_20():
    '''
    transformer of dataset: Caltech101_20
    :param  data['gist']  channel 1, shape=(n_simples, dims)
            data['hog'] channel 2, shape=(n_simples, dims)
            data['lbp']
            data['y'] label, shape=(n_simples, 1)

            ! classes = 20 (Y.max()==20)

    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                totalnum
    '''
    dataFile = os.path.join(root, 'Caltech101-20.mat')
    data = scio.loadmat(dataFile)

    X_1 = data['gist']
    X_2 = data['hog']
    X_3 = data['lbp']
    label = data['y']
    n_simples = data['y'].shape[0]

    X1_dims = X_1.shape[1]
    X2_dims = X_2.shape[1]
    X3_dims = X_3.shape[1]

    raw = np.hstack((X_1, X_2, X_3, label))
    totalnum = round((raw.shape[0]) / 2)  # train len
    raw = raw.tolist()
    train_set = np.array(random.sample(raw, totalnum))
    test_set = np.array(random.sample(raw, totalnum))

    S1_X = train_set[:, :X1_dims + X2_dims + X3_dims]
    S1_Y = train_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)
    S2_X = test_set[:, :X1_dims + X2_dims + X3_dims]
    S2_Y = test_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)

    return S1_X, S1_Y, S2_X, S2_Y, totalnum


def transformer_Mfeat():
    '''
    transformer of dataset: Mfeat
    :param  data['fac']  channel 1, shape=(n_simples, dims)
            data['fou'] channel 2, shape=(n_simples, dims)
            data['zer']
            data['y'] label, shape=(n_simples, 1)

            ! classes = 10 (Y.max()==10)

    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                totalnum
    '''
    dataFile = os.path.join(root, 'Mfeat.mat')
    data = scio.loadmat(dataFile)

    X_1 = data['fac']
    X_2 = data['fou']
    X_3 = data['zer']
    label = data['y']
    n_simples = data['y'].shape[0]

    X1_dims = X_1.shape[1]
    X2_dims = X_2.shape[1]
    X3_dims = X_3.shape[1]

    raw = np.hstack((X_1, X_2, X_3, label))
    totalnum = round((raw.shape[0]) / 2)  # train len
    raw = raw.tolist()
    train_set = np.array(random.sample(raw, totalnum))
    test_set = np.array(random.sample(raw, totalnum))

    S1_X = train_set[:, :X1_dims + X2_dims + X3_dims]
    S1_Y = train_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)
    S2_X = test_set[:, :X1_dims + X2_dims + X3_dims]
    S2_Y = test_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)

    return S1_X, S1_Y, S2_X, S2_Y, totalnum


def transformer_MSRC_v1():
    '''
    transformer of dataset: MSRC-v1
    :param  data['X'][0][0]  channel 1, shape=(n_simples, dims)
            data['X'][0][1]  channel 2, shape=(n_simples, dims)
            data['X'][0][2]  channel 3, shape=(n_simples, dims)

            data['Y'] label, shape=(n_simples, 1)

            ! classes = 6 (Y.max()==6)

    :return:    S1_X    训练集 X，X是多个视图的stack，    S1_X.shape = (simple_nums, dims_total)
                S1_Y    训练集 label ，                 S1_Y.shape = (simple_nums, 1)
                S2_X    测试集 X
                S2_Y    测试集 label
                totalnum
    '''
    dataFile = os.path.join(root, 'MSRC-v1.mat')
    data = scio.loadmat(dataFile)

    X = data['X']
    X_1 = X[0][0]
    X_2 = X[0][1]
    X_3 = X[0][2]

    label = data['Y']
    n_simples = data['Y'].shape[0]

    X1_dims = X_1.shape[1]
    X2_dims = X_2.shape[1]
    X3_dims = X_3.shape[1]

    raw = np.hstack((X_1, X_2, X_3, label))
    totalnum = round((raw.shape[0]) / 2)  # train len
    raw = raw.tolist()
    train_set = np.array(random.sample(raw, totalnum))
    test_set = np.array(random.sample(raw, totalnum))

    S1_X = train_set[:, :X1_dims + X2_dims + X3_dims]
    S1_Y = train_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)
    S2_X = test_set[:, :X1_dims + X2_dims + X3_dims]
    S2_Y = test_set[:, X1_dims + X2_dims + X3_dims].reshape(-1, 1)

    return S1_X, S1_Y, S2_X, S2_Y, totalnum



