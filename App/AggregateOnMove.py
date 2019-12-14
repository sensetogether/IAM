# -*- coding: utf-8 -*-

import time
import random
import numpy as np
import itertools as it
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Path of files
path = "/storage/emulated/0/Download/AggregateOnMove/"
record_aggregation = open(path + "record_aggregation.csv",'a')

def db_avg(db_list):
    po = sum([10**(db/10) for db in db_list])/len(db_list)
    return 10*np.log10(po)

# Get random index set from size and training/test ratio
def random_index(total_size, train_ratio):
    total_index = set(range(total_size))
    train_index = random.sample(total_index, int(total_size * train_ratio))
    test_index = total_index - set(train_index)
    return train_index, list(test_index)

# Devide a dataset into training and test set according to training/test ratio
def devide_file(file_name, train_ratio):
    lines = []
    with open(file_name,'r') as file_in:
        for line in file_in.readlines():
            lines.append(line.strip('\n'))
    index_train, index_test = random_index(len(lines), train_ratio)
    data_train = [lines[i] for i in index_train]
    data_test = [lines[i] for i in index_test]
    return data_train, data_test

# Read all data from dataset, averaging values on same location and time
def read_whole_data(data):
    xyzs, values = [], []
    for line in data:
        string = line.split(',')
        xyz = [int(string[1]), int(string[2]), int(string[3])]
        value = float(string[4])
        try: # Average the value in a same xyz slot
            values[xyzs.index(xyz)].append(value)
        except:
            xyzs.append(xyz)
            values.append([value])
    return xyzs, [db_avg(value) for value in values]

# Extract data to device id and data list
def read_device_data(data):
    deviceId, deviceData = [], []
    for line in data:
        string = line.split(',')
        device = string[0]
        data = [int(string[1]), int(string[2]), int(string[3]), float(string[4])]
        try:
            deviceData[deviceId.index(device)].append(data)
        except:
            deviceId.append(device)
            deviceData.append([data])
    return deviceId, deviceData

# Read the data of a device, averaging values on same location and time
def read_one_device(device_data):
    xyzs, values = [], []
    for line in device_data:
        xyz = [line[0], line[1], line[2]]
        value = line[3]
        try: # Average the value in a same xyz slot
            values[xyzs.index(xyz)].append(value)
        except:
            xyzs.append(xyz)
            values.append([value])
    return xyzs, [db_avg(value) for value in values]

# Convert the data to a tensor of given size
def data2tensor(xyzs, values, size):
    X, M = np.zeros(size), np.zeros(size)
    for xyz in xyzs:
        X[xyz[0],xyz[1],xyz[2]] = values[xyzs.index(xyz)]
        M[xyz[0],xyz[1],xyz[2]] = 1    
    return X, M

# Gaussian Process Regression on a tensor
def tensor_gpr(X, M):
    size = M.shape # = X.shape
    points = list(it.product(*[np.arange(size[0]),np.arange(size[1]),np.arange(size[2])]))
    xyzs = [points[i] for i in np.where(M.reshape(size[0]*size[1]*size[2])!=0)[0]]
    values = X[np.where(M!=0)]
    gp = GaussianProcessRegressor(kernel=Matern())
    gp.fit(xyzs, values)
    Xhat, S = gp.predict(points, return_std=True)
    return Xhat.reshape(size), S.reshape(size) ** 2

# Extrat the data renage from X, Y, Z coordinates
def xyzs2ranges(xyzs):
    ran, xyzs_a = {}, np.array(xyzs)
    ran['x_l'], ran['y_l'], ran['z_l'] = np.min(xyzs_a[:,0]), np.min(xyzs_a[:,1]), np.min(xyzs_a[:,2])
    ran['x_r'], ran['y_r'], ran['z_r'] = np.max(xyzs_a[:,0]), np.max(xyzs_a[:,1]), np.max(xyzs_a[:,2])
    return ran

# Check the overlap of two mask tensors
def has_overlap(M1, M2):
    size = M1.shape # = M2.shape
    points = list(it.product(*[np.arange(size[0]),np.arange(size[1]),np.arange(size[2])]))
    xyzs1 = [points[i] for i in np.where(M1.reshape(size[0]*size[1]*size[2])!=0)[0]]
    xyzs2 = [points[i] for i in np.where(M2.reshape(size[0]*size[1]*size[2])!=0)[0]]
    res1, res2 = xyzs2ranges(xyzs1), xyzs2ranges(xyzs2)
    return max(res1['x_l'], res2['x_l']) <= min(res1['x_r'], res2['x_r']) and max(res1['y_l'], res2['y_l']) <= min(res1['y_r'], res2['y_r']) and max(res1['z_l'], res2['z_l']) <= min(res1['z_r'], res2['z_r'])

# Compute the loss fuction between two tensors
def compute_cost(X1, M1, X2, M2):
    pos = np.where((M2 == 1) & (M1 == 0))
    return np.sum( (X2 - X1)[pos] ** 2) / M1[pos].shape[0]

# Merge two tensors into one tensor
def merge_tensor_gpr(X1, M1, n1, V1, X2, M2, n2, V2):
    # Initial
    X = np.zeros(X1.shape) # = X2.shape
    M = np.zeros(M1.shape) # = M2.shape
    # Mean ground truth
    X += (n1 * X1 + n2 * X2) * (M1 * M2) / (n1 + n2) 
    # Single ground truth
    X += X1 * (M1 *(1-M2))
    X += X2 * (M2 *(1-M1))
    # Mean inference
    w1 = 1
    w2 = 1
    X += ( w1 * X1/V1 + w2 * X2/V2) * ((1-M1) * (1-M2)) / ( w1 * 1/V1 + w2 * 1/V2) 
    # Merge M
    M += M1 * M2 + M1 *(1-M2) + M2 *(1-M1)
    n = n1 + n2
    return X, M, n, 1/(w1 * 1/V1 + w2 * 1/V2)

# Interpolation
def interpolation(X, M):
    return tensor_gpr(X, M)
    
# Aggregation
def aggregation(X1, M1, n1, V1, X2, M2, n2, V2):
    # Check the merge direction
    if (compute_cost(X1, M1, X2, M2) < compute_cost(X2, M2, X1, M1)):
        return True
    else:
        return False



#------------------------------------------------------------------------------



# Tensor size for one day
size = (100, 100, 24)

# Iteration on everyday
for day in range(1, 366):
    print("Reading day " + str(day))
    
    data_train, data_test = devide_file(path + "DataDay/Data (" + str(day) + ").csv", 1)        

    xyzs_train, values_train = read_whole_data(data_train)
    
    # Read device id and data of day
    ids, datas = read_device_data(data_train)

    # Opprtunistic aggragation
    x_list, m_list = [], []
    n_list, v_list = [], []
    aggregated = []
    aggr_time = 0
    for id_ in ids:
        xyzs, values = read_one_device(datas[ids.index(id_)])
        X, M = data2tensor(xyzs, values, size)
        X, V = interpolation(X, M)
        x_list.append(X)
        m_list.append(M)
        n_list.append(1)
        v_list.append(V)
    # Any pair of devices of the day
    for pair in it.permutations(ids, 2):        
        ix1, ix2 = ids.index(pair[0]), ids.index(pair[1])
        # Check the aggregation record
        if pair[0] not in aggregated and pair[1] not in aggregated:
            X1, M1, n1, V1, X2, M2, n2, V2 = x_list[ix1],  m_list[ix1], n_list[ix1], v_list[ix1], x_list[ix2], m_list[ix2], n_list[ix2], v_list[ix2]
            start = time.time()
            if has_overlap(M1, M2) and aggregation(X1, M1, n1, V1, X2, M2, n2, V2):
                print("Aggragating " +  pair[1] +" to " + pair[0])
                x_list[ix1], m_list[ix1], n_list[ix1], v_list[ix1] = merge_tensor_gpr(X1, M1, n1, V1, X2, M2, n2, V2)
                aggregated.append(pair[1])
            end = time.time()
            aggr_time = aggr_time + end - start
            
    print("Interpolation time: ", 0, "Aggregation time: ", aggr_time)
    entry = str(len(values_train)) + ',' + str(len(ids)) + ',' + str(len(aggregated)) + ',' + str(0) + ',' + str(aggr_time) + '\n'
    record_aggregation.write(entry)

record_aggregation.close()
