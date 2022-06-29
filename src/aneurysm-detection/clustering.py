from gettext import find
from operator import mod
import h5py
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import json

sys.setrecursionlimit(60000)
sys.path.append(os.path.join(sys.path[0], os.pardir, os.pardir)) 

# read file data
filename = os.path.join("/home/emi/Uni/Master/SoSe_22/MIP/Aneurysm-Detection/src/aneurysm-detection", "predictions/iteration2/A121_predictions.h5")
with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])

# mask convert to binary mask
newData = [[[0]*220]*256]*256
newData = np.squeeze(data, axis=0)  
newData[newData > 0.8] = 1
newData[newData <= 0.8] = 0

already_checked = np.full((256, 256, 220), False)

# variations of neighbor pixels
modify_array = [[ 1,0,0],
                [-1,0,0],
                [0,1,0],
                [0,-1,0],
                [0,0,1],
                [0,0,-1],
                [1,1,0],
                [1,-1,0],
                [-1,1,0],
                [-1,-1,0],
                [0,1,1],
                [0,1,-1],
                [0,-1,1],
                [0,-1,-1],
                [1,0,1],
                [1,0,-1],
                [-1,0,1],
                [-1,0,-1],
                [1,1,1],
                [1,1,-1],
                [1,-1,1],
                [1,-1,-1],
                [-1,1,1],
                [-1,1,-1],
                [-1,-1,1],
                [-1,-1,-1]]

# plot bounding box
def drawBoundingBox(ax, rrc):
    # z1 boundary
    ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b', label="a")
    ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b', label="b")
    ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='b', label="c")
    ax.plot(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]], color='b', label="d")

    # z2 plane boundary
    ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='b', label="e")
    ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='b', label="f")
    ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='b', label="g")
    ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='b', label="h")

    # z1 and z2 connecting boundaries
    ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='b', label="i")
    ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='b', label="j")
    ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='b', label="k")
    ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b', label="l")

    # ax.plot(rrc[0, 0], rrc[1, 0], rrc[2, 0], 'rx')
    # ax.plot(rrc[0, 5], rrc[1, 5], rrc[2, 5], 'rx')

    # ax.plot(rrc[0, 1], rrc[1, 1], rrc[2, 0], 'gx')
    # ax.plot(rrc[0, 6], rrc[1, 6], rrc[2, 6], 'gx')

    # ax.plot(rrc[0, 0], rrc[1, 0], rrc[2, 0], 'kx')
    # ax.plot(rrc[0, 2], rrc[1, 2], rrc[2, 2], 'kx')


# compute corners of bbox
def bbox_3D_2(centered_data):
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(centered_data[0, :]), np.max(centered_data[0, :]), np.min(centered_data[1, :]), np.max(centered_data[1, :]), np.min(centered_data[2, :]), np.max(centered_data[2, :])
    return xmin, xmax, ymin, ymax, zmin, zmax

def compute_bounding_box(ax, tmpData):
    cluster_data = np.where(tmpData == 1)
    #print(cluster_data)

    means = np.mean(cluster_data,axis=1)
    cov = np.cov(cluster_data)
    eval,evec = LA.eig(cov)

    centered_data = cluster_data - means[:,np.newaxis]
    np.allclose(LA.inv(evec), evec.T)
    aligned_coords = np.matmul(evec.T, centered_data)

    xmin, xmax, ymin, ymax, zmin, zmax =  bbox_3D_2(aligned_coords)

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                          [y1, y2, y2, y1, y1, y2, y2, y1],
                                                          [z1, z1, z1, z1, z2, z2, z2, z2]])
    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis] 

    drawBoundingBox(ax, rrc)

    v_m = np.zeros((4,3))
    v_m[0] = [(rrc[0,6] + rrc[0,0])/2, (rrc[1,6] + rrc[1,0])/2, (rrc[2,6] + rrc[2,0])/2]
    v_m[1] = [(rrc[0,5] + rrc[0,0])/2, (rrc[1,5] + rrc[1,0])/2, (rrc[2,5] + rrc[2,0])/2]
    v_m[2] = [(rrc[0,1] + rrc[0,6])/2, (rrc[1,1] + rrc[1,6])/2, (rrc[2,0] + rrc[2,6])/2]
    v_m[3] = [(rrc[0,2] + rrc[0,0])/2, (rrc[1,2] + rrc[1,0])/2, (rrc[2,2] + rrc[2,0])/2]

    v_a = np.around(v_m[1] - v_m[0], 4)
    v_b = np.around(v_m[2] - v_m[0], 4)
    v_c = np.around(v_m[3] - v_m[0], 4)

    # ax.plot([v_m[0][0],v_m[1][0]], [v_m[0][1],v_m[1][1]], [v_m[0][2], v_m[1][2]], c='red')
    # ax.plot([v_m[0][0],v_m[2][0]], [v_m[0][1],v_m[2][1]], [v_m[0][2], v_m[2][2]], c='orange')
    # ax.plot([v_m[0][0],v_m[3][0]], [v_m[0][1],v_m[3][1]], [v_m[0][2], v_m[3][2]], c='pink')

    middle_point = np.around(v_m[0], 4)

    text = {'position': middle_point.tolist(), 
            'object_oriented_bounding_box': {
                'extent': [1,1,1],
                'orthogonal_offset_vectors': [v_a.tolist(), v_b.tolist(), v_c.tolist()]
            }
            }

    return json.dumps(text)


def find_cluster_start(ax, x,y,z):
    tmp_array = np.zeros((256,256,220))
    tmp_array[x][y][z] = 1
    already_checked[x][y][z] = True

    tmp_array = recursive_cluster(tmp_array, x,y,z)

    if(np.count_nonzero(tmp_array) > 1):
        json_bounding_box = compute_bounding_box(ax, tmp_array)
        return json_bounding_box
    return ""

def recursive_cluster(tmp_array, x,y,z):
    for variant in modify_array:
        newx = x + variant[0]
        newy = y + variant[1]
        newz = z + variant[2]

        if newx >= 0 and newx <= 255 and newy >= 0 and newy <= 255 and newz >= 0 and newz <= 219:
            if newData[newx][newy][newz] == 1 and not already_checked[newx][newy][newz]:            
                tmp_array[newx][newy][newz] = 1
                already_checked[newx][newy][newz] = True

                tmp_array = recursive_cluster(tmp_array, newx, newy, newz)

    return tmp_array

### PLOT
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

json_final = '{"dataset_id": "XX", "processing_time_in_seconds": 1000, "candidates" : ['
jsons = []

for z in range(0,220):
    for x in range(0,255):
        for y in range(0,255):
            if newData[x][y][z] == 1 and not already_checked[x][y][z]:
                json_cluster = find_cluster_start(ax, x,y,z)
                if json_cluster != "":
                    jsons.append(json_cluster)

for single_json in jsons:
    json_final += single_json
    json_final += ','

json_final += "]}"
result = json_final

with open('bounding_box.json', 'w') as outputfile:
    outputfile.write(result)
    
original_data = np.where(newData == 1)
ax.scatter(original_data[0], original_data[1], original_data[2], c='green')
plt.show()