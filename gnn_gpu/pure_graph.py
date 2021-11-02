import pandas as pd
import json

USER_OBJECTS = pd.read_csv("/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/people_vk_pure_0.csv", usecols = ["id", "friends"])
data_file_name = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/all_connections.json"
data_file_name2 = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/some_connections.json"

id_friends = dict([(int(user[1]["id"]), [int(id) for id in user[1]["friends"][1:-1].split(", ")]) for user in USER_OBJECTS.iterrows()])

answer = dict([(int(user[1]["id"]), [int(id) for id in user[1]["friends"][1:-1].split(", ")]) for user in USER_OBJECTS.iterrows()])

for id in id_friends:
    for node in id_friends[id]:
        if node not in answer:
            answer[node] = []
        answer[node].append(id)


with open(data_file_name, "w") as outfile:
    json.dump(answer, outfile)




answer2 = dict([(int(user[1]["id"]), [int(id) for id in user[1]["friends"][1:-1].split(", ")]) for user in USER_OBJECTS.iterrows()])
for id in id_friends:
    for node in id_friends[id]:
        if node not in id_friends:
            answer2[id].remove(node)

with open(data_file_name2, "w") as outfile:
    json.dump(answer2, outfile)