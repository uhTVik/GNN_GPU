import json
import pandas as pd
import facebook

params_json_file_name = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/params.json"
graph_depth = 0

data_file_name = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/people_fb_"+str(graph_depth)+".csv"

with open(params_json_file_name) as params_json:
    params = json.load(params_json)

graph = facebook.GraphAPI(params["fb"]["token"], version="2.12")
print(graph)
friends = graph.get_connections(id='me', connection_name='friends')
print(friends)
for friend in friends['data']:
    print("{0} has id {1}".format(friend['name'].encode('utf-8'), friend['id']))