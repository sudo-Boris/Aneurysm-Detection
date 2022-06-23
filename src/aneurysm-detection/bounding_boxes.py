import h5py
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


# read file data
filename = "predictions/iteration2/A121_predictions.h5"
with h5py.File(filename, "r") as f:
    a_group_key = list(f.keys())[0]
    data = list(f[a_group_key])

# mask convert to binary mask
newData =[[[0]*220]*256]*256
newData = np.squeeze(data, axis=0)  
newData[newData > 0.8] = 1
newData[newData <= 0.8] = 0

original_data = np.where(newData == 1)
#print(list(original_data))
print(original_data)


means = np.mean(original_data,axis=1)
cov = np.cov(original_data)
eval,evec = LA.eig(cov)

centered_data = original_data - means[:,np.newaxis]
np.allclose(LA.inv(evec), evec.T)
aligned_coords = np.matmul(evec.T, centered_data)

xmin, xmax, ymin, ymax, zmin, zmax =  bbox_3D_2(aligned_coords)
print(xmin, xmax, ymin, ymax, zmin, zmax)

rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                      [y1, y2, y2, y1, y1, y2, y2, y1],
                                                      [z1, z1, z1, z1, z2, z2, z2, z2]])
realigned_coords = np.matmul(evec, aligned_coords)
realigned_coords += means[:, np.newaxis]
rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
rrc += means[:, np.newaxis] 


### PLOT
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# plot aneurysm
ax.scatter(realigned_coords[0], realigned_coords[1], realigned_coords[2], c='green')
drawBoundingBox(ax, rrc)

plt.show()