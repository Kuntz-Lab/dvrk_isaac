import numpy as np
import pickle
import os
import pandas as pd

def filter_data(result_dir, prim_name, obj_type, inside, range_data):
    chamfer_data = []
    for i in range(range_data[0], range_data[1]):
        if inside:
            file_name = os.path.join(result_dir, obj_type, "inside", f"{prim_name}_{str(i)}.pickle")
        else:
            file_name = os.path.join(result_dir, obj_type, "outside", f"{prim_name}_{str(i)}.pickle")
        if os.path.isfile(file_name):
            with open(file_name, 'rb') as handle:
                data = pickle.load(handle)
            chamfer_data.extend(data)


    return chamfer_data


result_dir = "/home/baothach/shape_servo_data/evaluation/chamfer_results"
prim_name = "cylinder"



inside_data = {}
outside_data = {}

range_data = [0,10]
recording_path = "/home/baothach/shape_servo_data/evaluation/visualization/plot_data_1/refined"
  


with open(os.path.join(recording_path, "inside", "result.pickle"), 'rb') as handle:
    inside_data = pickle.load(handle) 
with open(os.path.join(recording_path, "outside", "result.pickle"), 'rb') as handle:
    outside_data = pickle.load(handle) 


datas = []
object_names = []
categories = []
for prim_name in ["cylinder", "box", "hemis"]:
    for i, obj_type in enumerate(['1k', '5k', '10k']):
        chamfers = inside_data[prim_name][i]
        datas.extend(chamfers)
        object_names.extend([f"{prim_name} {obj_type}"]*len(chamfers))
        categories.extend(["inside"]*len(chamfers))

        chamfers = outside_data[prim_name][i]
        datas.extend(chamfers)
        object_names.extend([f"{prim_name} {obj_type}"]*len(chamfers))
        categories.extend(["outside"]*len(chamfers))


df =  pd.DataFrame()
df["chamfer"] = datas
df["obj name"] = object_names
df["Category"] = categories
# print(df)

with open(os.path.join(recording_path, "result.pickle"), 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL) 

