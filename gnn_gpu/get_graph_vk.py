import vk_api
import json
import pandas as pd

params_json_file_name = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/params.json"
graph_depth = 0

data_file_name = "/Users/aleksandrukhatov/Documents/Projects/GNN_GPU/data/people_vk_pure_"+str(graph_depth)+".csv"

with open(params_json_file_name) as params_json:
    params = json.load(params_json)

vk_session = vk_api.VkApi(params["vk"]["login"], params["vk"]["password"])
vk_session.auth()
super_id = int(vk_session.token["user_id"])

vk = vk_session.get_api()

friends = vk.friends.get(user_id='')
user_ids = set(friends['items'])
user_ids.add(super_id)
friends_dict = dict()
friends_dict[super_id] = user_ids

new_user_ids = []
print("depth = 0: ", len(user_ids))
for i in range(graph_depth):
    n = len(user_ids)
    for j, user_id in enumerate(user_ids):
        percent = int(100 * j / n)
        if percent % 5 == 0:
            print(str(percent) + "%")
        try:
            friends = vk.friends.get(user_id=user_id)
            cur_user_ids = friends['items']
            new_user_ids = new_user_ids + cur_user_ids
            friends_dict[user_id] = cur_user_ids
        except Exception as e:
            print(user_id, e)
    user_ids.update(set(new_user_ids))
    print("depth = " +  str(i+1) + ": " + str(len(user_ids)))
    print(len(user_ids))

user_ids_list = list(user_ids)
fields = "verified, sex, bdate, city, country, home_town, education, universities, schools, followers_count, occupation, relatives, relation, personal, career"
# keys = fields.split(", ")
USER_OBJECTS = vk.users.get(user_ids=user_ids_list)

USER_OBJECTS_filtered = list(filter(lambda USER_OBJECT: USER_OBJECT['first_name'] != 'DELETED' and "deactivated" not in USER_OBJECT and USER_OBJECT['is_closed'] == False, USER_OBJECTS))

columns_to_drop = ["first_name", "is_closed", "last_name", "relation", "career", "can_access_closed", "verified", "university", "faculty", "education_form", "relation_partner", "personal", "home_town", "university_name", "faculty_name", "occupation"]
n = len(USER_OBJECTS_filtered)
for i, user_object_i in enumerate(USER_OBJECTS_filtered):
    percent = int(100 * i / n)
    if percent % 5 == 0:
        print(str(percent) + "%")
    if user_object_i["id"] in friends_dict:
        user_object_i["friends"] = friends_dict[user_object_i["id"]]
    else:
        user_object_i["friends"] = vk.friends.get(user_id=user_object_i["id"])["items"]
    if "city" in user_object_i:
        user_object_i["city"] = user_object_i["city"]["title"]
    if "country" in user_object_i:
        user_object_i["country"] = user_object_i["country"]["title"]
    if "universities" in user_object_i:
        user_object_i["universities"] = list(set(uni["name"] for uni in user_object_i["universities"]))
    if "schools" in user_object_i:
        user_object_i["schools"] = list(set(sch["name"] for sch in user_object_i["schools"]))
    if "relatives" in user_object_i:
        user_object_i["relatives"] = list(set(rel["id"] for rel in user_object_i["relatives"]))


dataframe = pd.DataFrame.from_dict(USER_OBJECTS_filtered)
dataframe = dataframe.drop(columns=columns_to_drop)
dataframe.to_csv(data_file_name)
