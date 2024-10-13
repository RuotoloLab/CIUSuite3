"""
Author: Carolina Rojas Ramirez
CIUSUite3 Testing: 3D LDA Graphing
Oct 12th, 2024
"""
import os
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

classifiedfile = filedialog.askopenfilename(title = "Classified File")
print(classifiedfile)
outfolder = os.path.dirname(classifiedfile)
os.chdir(outfolder)
print(outfolder)
##LD 1 (linear discriminant dimension 1)	LD 2 (linear discriminant dimension 2)	LD 3 (linear discriminant dimension 3)	LD 4 (linear discriminant dimension 4)	Class Prediction

classified_df = pd.read_csv(classifiedfile)

# Don't read last row
nolastrow = classified_df[:-1]
# print(nolastrow)

clusteridx = []

clusterlables = list(set(nolastrow["Class Prediction"]))

print(clusterlables)
clusterlabelsidx_dict = {}

for clustername in clusterlables:
    clusterlabelsidx_dict[clustername] = index = clusterlables.index(clustername)


clusterlistfrograph = []
for clustername in nolastrow["Class Prediction"]:
    # print(clustername)
    clusterlistfrograph.append(clusterlabelsidx_dict[clustername])


fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection='3d')
ax.grid()

for cluster in clusterlables:
    clustercoor = classified_df[classified_df["Class Prediction"] == cluster]
    x = clustercoor["LD 1 (linear discriminant dimension 1)"]
    y = clustercoor["LD 2 (linear discriminant dimension 2)"]
    z = clustercoor["LD 3 (linear discriminant dimension 3)"]

    ax.scatter(x, y, z, s=80, alpha=0.4, label=cluster)





# x = ldone
# y = ldtwo
# z = ldthree
#
# # ax.plot3D(x, y, z)
# ax.scatter(x, y, z, c=clusterlistfrograph, s = 80, alpha = 0.4, label=clusterlabelsidx_dict)
ax.set_title('ctrl_acid_base_Heat_37_FreezeThaw_4SubCl_Results')
# Set axes label
ax.set_xlabel('LD1', labelpad=20)
ax.set_ylabel('LD2', labelpad=20)
ax.set_zlabel('LD3', labelpad=20)
ax.legend()

plt.savefig("LDA_CIUSUIte3tes.pdf")
plt.show()

