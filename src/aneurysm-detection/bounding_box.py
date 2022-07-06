import glob
import logging
from math import nan
import os
from pathlib import Path
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

sys.setrecursionlimit(50000)

already_checked = np.full((256, 256, 220), False)

modify_array = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [-1, -1, 0],
    [0, 1, 1],
    [0, 1, -1],
    [0, -1, 1],
    [0, -1, -1],
    [1, 0, 1],
    [1, 0, -1],
    [-1, 0, 1],
    [-1, 0, -1],
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1],
]


def get_pred_for_case(case: str, iteration: int, threshold: int = 0.9) -> np.ndarray:
    file = os.path.join(
        "/home/borismeinardus/Aneurysm-Detection/data/predictions/exam",
        "iteration{}/{}_predictions.h5".format(iteration, case),
    )

    with h5py.File(file, "r") as f:
        pred = f["predictions"][:]

    pred = np.squeeze(pred, axis=0)
    if pred.shape == (220, 256, 256):
        # reshape ds from z, x, y to x, y, z
        pred = np.moveaxis(pred, 0, -1)
    pred[pred > threshold] = 1
    pred[pred <= threshold] = 0

    print("shape {}: ".format(case) + str(pred.shape))
    return pred


def drawBoundingBox(ax, rrc):
    # z1 boundary
    ax.plot(rrc[0, 0:2], rrc[1, 0:2], rrc[2, 0:2], color="b", label="a")
    ax.plot(rrc[0, 1:3], rrc[1, 1:3], rrc[2, 1:3], color="b", label="b")
    ax.plot(rrc[0, 2:4], rrc[1, 2:4], rrc[2, 2:4], color="b", label="c")
    ax.plot(rrc[0, [3, 0]], rrc[1, [3, 0]], rrc[2, [3, 0]], color="b", label="d")

    # z2 plane boundary
    ax.plot(rrc[0, 4:6], rrc[1, 4:6], rrc[2, 4:6], color="b", label="e")
    ax.plot(rrc[0, 5:7], rrc[1, 5:7], rrc[2, 5:7], color="b", label="f")
    ax.plot(rrc[0, 6:], rrc[1, 6:], rrc[2, 6:], color="b", label="g")
    ax.plot(rrc[0, [7, 4]], rrc[1, [7, 4]], rrc[2, [7, 4]], color="b", label="h")

    # z1 and z2 connecting boundaries
    ax.plot(rrc[0, [0, 4]], rrc[1, [0, 4]], rrc[2, [0, 4]], color="b", label="i")
    ax.plot(rrc[0, [1, 5]], rrc[1, [1, 5]], rrc[2, [1, 5]], color="b", label="j")
    ax.plot(rrc[0, [2, 6]], rrc[1, [2, 6]], rrc[2, [2, 6]], color="b", label="k")
    ax.plot(rrc[0, [3, 7]], rrc[1, [3, 7]], rrc[2, [3, 7]], color="b", label="l")


def find_cluster_start(pred, x, y, z):
    """Initialize the recursive function to find the cluster starting with a given coordinate

    Args:
        pred (np.ndarray): whole prediction for which to find cluster for.
        x (int): x coordinate
        y (int): y coordinate
        z (int): z coordinate

    Returns:
        cluster (np.ndarray): the cluster that was found starting at (x, y, z)
    """
    tmp_array = np.zeros((256, 256, 220))
    tmp_array[x][y][z] = 1
    already_checked[x][y][z] = True

    cluster = recursive_cluster(pred, tmp_array, x, y, z)

    return cluster


def recursive_cluster(pred, tmp_array, x, y, z):
    """Recursive function to find the cluster starting with a given coordinate

    Args:
        pred (np.ndarray): whole prediction for which to find cluster for.
        tmp_array (np.ndarray): current cluster
        x (int): x coordinate
        y (int): y coordinate
        z (int): z coordinate

    Returns:
        tmp_array (np.ndarray): current cluster
    """
    for variant in modify_array:
        newx = x + variant[0]
        newy = y + variant[1]
        newz = z + variant[2]

        if (
            newx >= 0
            and newx <= 255
            and newy >= 0
            and newy <= 255
            and newz >= 0
            and newz <= 219
        ):
            if pred[newx][newy][newz] == 1 and not already_checked[newx][newy][newz]:
                tmp_array[newx][newy][newz] = 1
                already_checked[newx][newy][newz] = True

                tmp_array = recursive_cluster(pred, tmp_array, newx, newy, newz)

    return tmp_array


def bbox_3D_2(centered_data):
    xmin, xmax, ymin, ymax, zmin, zmax = (
        np.min(centered_data[0, :]),
        np.max(centered_data[0, :]),
        np.min(centered_data[1, :]),
        np.max(centered_data[1, :]),
        np.min(centered_data[2, :]),
        np.max(centered_data[2, :]),
    )
    return xmin, xmax, ymin, ymax, zmin, zmax


def compute_bounding_box(ax, data):
    """Compute object oriented bounding box for one clustering

    Args:
        ax (fig.add_subplot(projection="3d")): axes to which to add current bounding box.
        data (np.ndarray): the current cluster of data (Aneurysm) for which to compute the bounding box.

    Returns:
        rrc (np.ndarray): bounding box.
    """
    cluster_data = np.where(data == 1)

    means = np.mean(cluster_data, axis=1)
    cov = np.cov(cluster_data)
    eval, evec = np.linalg.eig(cov)

    centered_data = cluster_data - means[:, np.newaxis]
    np.allclose(np.linalg.inv(evec), evec.T)
    aligned_coords = np.matmul(evec.T, centered_data)

    xmin, xmax, ymin, ymax, zmin, zmax = bbox_3D_2(aligned_coords)

    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array(
        [
            [x1, x1, x2, x2, x1, x1, x2, x2],
            [y1, y2, y2, y1, y1, y2, y2, y1],
            [z1, z1, z1, z1, z2, z2, z2, z2],
        ]
    )
    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]
    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis]

    if ax is not None:
        drawBoundingBox(ax, rrc)

    return rrc


def get_bounding_boxes(pred, viz=False):
    """Get bounding boxes from prediction.

    Args:
        pred (np.ndarray): binary segmentation map from prediction.
        viz (bool, optional): determine if you want to vizualise/ plot the prediction and respective bounding boxes. Defaults to False.

    Returns:
        list[np.ndarray]: list of bounding boxes.
    """
    if viz:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        original_data = np.where(pred == 1)
        ax.scatter(original_data[0], original_data[1], original_data[2], c="green")
    else:
        ax = None

    bboxes = []

    for z in range(0, 220):
        for x in range(0, 256):
            for y in range(0, 256):
                if pred[x][y][z] == 1 and not already_checked[x][y][z]:
                    cluster = find_cluster_start(pred, x, y, z)
                    non_zeros = np.count_nonzero(cluster)
                    if non_zeros > 1:
                        bboxes.append(compute_bounding_box(ax, cluster))

    if viz:
        plt.show()

    return bboxes


def get_candidates_for_json(rrc: np.ndarray):
    """Compute middle point and orthogonal offset vectors for a candidate bounding box \
        and return in correct json format.

    Args:
        rrc (np.ndarray): Ooe bounding box in the form of a numpy array.

    Returns:
        dict: json candidates entry for current bounding box in the form of a dictionary.
    """
    v_m = np.zeros((4, 3))
    v_m[0] = [
        (rrc[0, 6] + rrc[0, 0]) / 2,
        (rrc[1, 6] + rrc[1, 0]) / 2,
        (rrc[2, 6] + rrc[2, 0]) / 2,
    ]
    v_m[1] = [
        (rrc[0, 5] + rrc[0, 0]) / 2,
        (rrc[1, 5] + rrc[1, 0]) / 2,
        (rrc[2, 5] + rrc[2, 0]) / 2,
    ]
    v_m[2] = [
        (rrc[0, 1] + rrc[0, 6]) / 2,
        (rrc[1, 1] + rrc[1, 6]) / 2,
        (rrc[2, 0] + rrc[2, 6]) / 2,
    ]
    v_m[3] = [
        (rrc[0, 2] + rrc[0, 0]) / 2,
        (rrc[1, 2] + rrc[1, 0]) / 2,
        (rrc[2, 2] + rrc[2, 0]) / 2,
    ]

    v_a = np.around(v_m[1] - v_m[0], 4)
    v_b = np.around(v_m[2] - v_m[0], 4)
    v_c = np.around(v_m[3] - v_m[0], 4)

    middle_point = np.around(v_m[0], 4)

    return {
        "position": middle_point.tolist(),
        "object_oriented_bounding_box": {
            "extent": [1, 1, 1],
            "orthogonal_offset_vectors": [v_a.tolist(), v_b.tolist(), v_c.tolist()],
        },
    }


def bboxes_to_json(case, processing_time, bboxes):
    """Brings the bounding boxes to the desired json format.

    Args:
        case (str): case name.
        processing_time (time): time it took to compute all bounding boxes for case.
        bboxes (list[np.ndarray]): list of bounding boxes.

    Returns:
        json_output (dict): dict with correct format for json output
    """
    json_output = {
        "dataset_id": case,
        "processing_time_in_seconds": processing_time,
    }
    candidates = []
    for bbox in bboxes:
        candidates.append(get_candidates_for_json(bbox))
    json_output["candidates"] = candidates

    return json_output


def get_cases(path):
    """Get all cases in the given path.

    Args:
        path (str): path to the folder containing the cases.

    Returns:
        list[str]: list of cases.
    """

    def get_file_name(file_path=None):
        # Some names are Axxx_L or Axxx_R. Those have to be included, otherwise names are onlz Axxx.
        file_name = os.path.basename(file_path)[:6]
        if file_name[4:] == "_L" or file_name[4:] == "_R" or file_name[4:] == "_M":
            return file_name
        return file_name[0:4]

    h5_files = sorted(glob.glob(os.path.join(path, "*")))

    cases = []
    for file in h5_files:
        cases.append(get_file_name(file))
    return cases


def main():
    iteration = 5
    cases = ["A121"]  # 'A120' 'A121' 'A123' 'A124' 'A126' 'A127' 'A129'
    threshold = 0.5
    viz = False

    predictions_path = os.path.join(
        "/home/borismeinardus/Aneurysm-Detection/data/predictions/exam",
        f"iteration{iteration}",
    )

    cases = get_cases(predictions_path)
    cases.remove("A036")  # A036 has wrong shape! (228, 256, 256)
    cases.remove("A104")  # A104 leads to segmentation fault. Too deep recursion...
    cases.remove("A144_L")  # A144_L has wrong shape! (221, 256, 256)
    cases.remove("A144_M")  # A144_L has wrong shape! (221, 256, 256)
    # A147 has wrong shape! (221, 256, 256)
    # A149 has wrong shape! (221, 256, 256)
    json_output = {
        "username": "Bagel",
        "task_1_results": [],
    }

    # sys.setrecursionlimit(40000)

    cases = ["A104"]

    for case in cases:
        pred = get_pred_for_case(case, iteration, threshold)
        if pred.shape != (256, 256, 220):
            print(f"{case} has wrong shape!")
            continue
        start = time.time()
        bboxes = get_bounding_boxes(pred, viz)
        processing_time = time.time() - start
        json_output["task_1_results"].append(
            bboxes_to_json(case, processing_time, bboxes)
        )
        print("Got bounding boxes for case {}".format(case))

    with open("bounding_box0.json", "w") as f:
        json.dump(json_output, f)

    if viz:
        plt.show()


if __name__ == "__main__":
    main()
