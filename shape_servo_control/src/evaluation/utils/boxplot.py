import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11})  # change font to size larger


recording_path = "/home/baothach/shape_servo_data/evaluation/visualization/plot_data_1/refined"


with open(os.path.join(recording_path, "result.pickle"), 'rb') as handle:
    all_data = pickle.load(handle) 
    # print(type(all_data))

prim_name = "box"  

ax=sns.boxplot(x="obj name", y="chamfer",hue='Category',data=all_data, showfliers = False)

plt.title('Experiment Results Over Multiple Objects', fontsize=40)
plt.xlabel('Object',fontsize=40)
plt.ylabel('Chamfer Distance (m)', fontsize=40)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32, rotation=90)
# print(ax.get_xticklabels())
plt.subplots_adjust(bottom=0.15) # Make x axis label (Object) fit
ax.tick_params(axis='both', which='major', labelsize=32)
ax.set_xticklabels(('C1','C5','C10','B1','B5','B10','H1','H5','H10'))
plt.legend(prop={'size': 38})
plt.show()



