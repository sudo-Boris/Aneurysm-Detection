from gettext import find
from operator import mod
import h5py
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

sys.setrecursionlimit(60000)

# read file data
filename = "predictions/iteration2/A120_predictions.h5"
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
    ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color='b')
    ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color='b')
    ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color='b')
    ax.plot(rrc[0, [3,0]], rrc[1, [3,0]], rrc[2, [3,0]], color='b')

    # z2 plane boundary
    ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color='b')
    ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color='b')
    ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color='b')
    ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color='b')

    # z1 and z2 connecting boundaries
    ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color='b')
    ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color='b')
    ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color='b')
    ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color='b')

def bbox_3D_2(centered_data):
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(centered_data[0, :]), np.max(centered_data[0, :]), np.min(centered_data[1, :]), np.max(centered_data[1, :]), np.min(centered_data[2, :]), np.max(centered_data[2, :])
    return xmin, xmax, ymin, ymax, zmin, zmax

def compute_bounding_box(ax, tmpData):
    cluster_data = np.where(tmpData == 1)
    print(cluster_data)

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


def find_cluster_start(ax, x,y,z):
    tmp_array =[[[0]*220]*256]*256
    tmp_array[x][y][z] = 1
    already_checked[x][y][z] = True

    tmp_array = recursive_cluster(tmp_array, x,y,z)
    tmp_converted = np.array(tmp_array, dtype=object)

    compute_bounding_box(ax, tmp_converted)

def recursive_cluster(tmp_array, x,y,z):
    for variant in modify_array:
        newx = x + variant[0]
        newy = y + variant[1]
        newz = z + variant[2]

        if newx >= 0 and newx <= 255 and newy >= 0 and newy <= 255 and newz >= 0 and newz <= 219:
            if newData[newx][newy][newz] == 1 and not already_checked[newx][newy][newz]:
                #print("including point", newx, newy, newz, "with value", newData[newx][newy][newz])
                print("NEW DATA: ", newData[newx][newy][newz], " checked", already_checked[newx][newy][newz])
                print("going in")
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

for z in range(0,220):
    for x in range(0,255):
        for y in range(0,255):
            if newData[x][y][z] == 1 and not already_checked[x][y][z]:
                print("new cluster found at", x, y, z, "VAL", newData[x][y][z])
                find_cluster_start(ax, x,y,z)
                

original_data = np.where(newData == 1)
ax.scatter(original_data[0], original_data[1], original_data[2], c='green')
plt.show()