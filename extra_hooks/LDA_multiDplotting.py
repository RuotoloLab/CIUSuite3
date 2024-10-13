from tkinter import filedialog
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

classifiedfile = filedialog.askopenfilename(title = "Classified File")
print(classifiedfile)
##LD 1 (linear discriminant dimension 1)	LD 2 (linear discriminant dimension 2)	LD 3 (linear discriminant dimension 3)	LD 4 (linear discriminant dimension 4)	Class Prediction

classified_df = pd.read_csv(classifiedfile)

# Don't read last row
nolastrow = classified_df[:-1]
# print(nolastrow)

ldone = np.array(nolastrow["LD 1 (linear discriminant dimension 1)"])
ldtwo = np.array(nolastrow["LD 2 (linear discriminant dimension 2)"])
ldthree = np.array(nolastrow["LD 3 (linear discriminant dimension 3)"])
ldfour = np.array(nolastrow["LD 4 (linear discriminant dimension 4)"])

clusteridx = []

clusterlables = list(set(nolastrow["Class Prediction"]))

print(clusterlables)
clusterlabelsidx_dict = {}

for clustername in clusterlables:
    clusterlabelsidx_dict[clustername] = index = clusterlables.index(clustername)


clusterlistfrograph = []
for clustername in nolastrow["Class Prediction"]:
    print(clustername)
    clusterlistfrograph.append(clusterlabelsidx_dict[clustername])

for cluster in clusterlables:
    clustercoor = classified_df[classified_df["Class Prediction"] == cluster]
    print(clustercoor["LD 1 (linear discriminant dimension 1)"])




# fig = plt.figure(figsize = (8,8))
# ax = plt.axes(projection='3d')
# ax.grid()
# x = ldone
# y = ldtwo
# z = ldthree
#
# # ax.plot3D(x, y, z)
# ax.scatter(x, y, z, c=clusterlistfrograph, s = 80, alpha = 0.4, label=clusterlabelsidx_dict)
# ax.set_title('3D Parametric Plot')
# # Set axes label
# ax.set_xlabel('LD1', labelpad=20)
# ax.set_ylabel('LD2', labelpad=20)
# ax.set_zlabel('LD3', labelpad=20)
# ax.legend()
#
# plt.show()