import pandas as pd
import json
import numpy as np
import random
import scipy
from scipy.sparse import coo_matrix



USER_OBJECTS = pd.read_csv("/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/people_vk_pure_0.csv", usecols = ["id", "friends", "schools"])

data_file_name_some = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/some_connections.json"
users = json.load(open(data_file_name_some, "r"))

target_school = "Физико-техническая школа (ФТШ)"
all_users_list = []
for i in users:
    all_users_list = all_users_list + users[i]

all_users_list = list(set(all_users_list))

dict_users_fake_to_real = dict(USER_OBJECTS["id"])
dict_users_real_to_fake = {}
for i in dict_users_fake_to_real:
    dict_users_real_to_fake[dict_users_fake_to_real[i]] = i

dict_friends = dict(USER_OBJECTS["friends"])
dict_schools = dict(USER_OBJECTS["schools"])

s = (len(dict_friends), len(dict_friends))
y = np.zeros(len(dict_friends))
adj_matrix = np.zeros(s)

row_ = []
col_ = []
data_ = []

for i in dict_friends:
    friends_cur = [int(f) for f in (dict_friends[i][1:-1]).split(",")]
    for f in friends_cur:
        if f in all_users_list:
            adj_matrix[i, dict_users_real_to_fake[f]] = 1
            adj_matrix[dict_users_real_to_fake[f], i] = 1
            row_.append(i)
            col_.append(dict_users_real_to_fake[f])
            data_.append(1)
            row_.append(dict_users_real_to_fake[f])
            col_.append(i)
            data_.append(1)

    y[i] = int(target_school in str(dict_schools[i]))

node_types = []
for i in range(len(y)):
    node_types.append(random.choice([1, 1, 1, 1, 1, 1, 1, 2, 2, 3]))

node_ids = np.array(list(range(0, len(y))))
feature = np.squeeze(np.asarray(adj_matrix))
node_types = np.array(node_types)
label = y

data = {}

data["node_ids"] = node_ids
data["feature"] = feature
data["node_types"] = node_types
data["label"] = label
# G_masked = np.ma.masked_values(adj_matrix, 0)

row = np.array(row_)
col = np.array(col_)
data_ = np.array(data_)
adj_matrix_sparse = coo_matrix((data_, (row, col)), shape=(len(y),  len(y)))

np.savez("../data/VK/raw/vk_data.npz", feature=feature, node_ids=node_ids, node_types=node_types, label=label)
scipy.sparse.save_npz("../data/VK/raw/vk_graph.npz", adj_matrix_sparse)


