import os
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pyvista as pv
from ipywidgets import widgets


def mask_overlay(image, mask, color=(0.0, 1.0, 0.0), weight=0.5):
    """
    Helper function to visualize mask on the top of the aneurysm
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = image * weight + mask * (
        1 - weight
    )  # cv2.addWeighted(image, 1 - weight, mask, weight, 0.,  dtype=cv2.CV_32F)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


class ImageSlicer(object):
    def __init__(self, ax, data_path, case):
        self.ax = ax

        # load NIFTI image & mask
        self.image_nib = nib.load(data_path / "{}_orig.nii.gz".format(case))
        self.mask_nib = nib.load(data_path / "{}_masks.nii.gz".format(case))

        # convert to numpy array
        self.image_np = self.image_nib.get_fdata()
        self.mask_np = self.mask_nib.get_fdata()

        # get number of slices
        _, _, self.slices = self.image_np.shape
        self.ind = self.slices // 2

        # plot image with mask overlay
        self.image_plt = self.ax.imshow(self.overlay)
        self._draw()

    @property
    def overlay(self):

        # get image and mask slice
        image = self.image_np[:, :, self.ind]
        image = image / np.max(image)
        image = np.dstack((image, image, image))
        mask = self.mask_np[:, :, self.ind]

        # create masked overlay
        return mask_overlay(image, mask)

    def onscroll(self, event):

        # get new slice number
        self.ind = event["new"]
        self.update()

    def update(self):

        # draw overlay
        self.image_plt.set_data(self.overlay)
        self._draw()

    def _draw(self):
        self.image_plt.axes.figure.canvas.draw()


def plot3d(data_path, case):

    #
    figure, ax = plt.subplots(1, 1)
    ax.set_title(case)
    tracker = ImageSlicer(ax, data_path, case)

    #
    int_slider = widgets.IntSlider(
        value=tracker.ind,
        min=0,
        max=tracker.slices,
        step=1,
        description="Slice",
        continuous_update=True,
    )
    int_slider.observe(tracker.onscroll, "value")

    return figure, int_slider


def plotMesh(data_path, case):
    # create a mesh and identify some scalars you wish to plot
    mesh = pv.read(data_path / "{}_vessel.stl".format(case))
    z = mesh.points[:, 2]

    # Plot using the ITKplotter
    pl = pv.PlotterITK()
    pl.add_mesh(mesh, smooth_shading=True)
    pl.show(True)
