import glob
import logging
import os
from datetime import datetime
from re import L

import h5py
import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


class Nifty_Converter:
    def __init__(self, filepaths_raw=None, filepaths_masks=None) -> None:
        assert filepaths_raw is not None, "No path to raw data provided."
        assert filepaths_masks is not None, "No path to mask/ label data provided."
        self.filepaths = zip(filepaths_raw, filepaths_masks)

    def nifty_to_h5(self, target_path=None):
        """Convert nifty images into h5 format for training with 3DUNet. Save them to a target path.

        Args:
            target_path (Path): Path where the hd5f files are to be saved. Defaults to None.
        """

        assert target_path is not None, "No target path provided."

        for raw_file, label_file in self.filepaths:
            name = self.get_file_name(raw_file)
            raw, label = self.nifty_to_numpy(raw_file, label_file)
            self.handle_h5(target_path, name, mode="a", raw=raw, label=label)
            logging.info("Created file {}".format(name + ".h5"))

    def handle_h5(
        self,
        target_path=None,
        name=None,
        mode="r",
        raw=np.zeros(256, 256, 220),
        label=np.zeros(256, 256, 220),
    ):
        """Create and return a h5 file to target path with respective naming. Store raw and label data in the created h5 file.

        Args:
            target_path (Path): Path where the hd5f files are to be saved. Defaults to None.
            name (str, optional): Name of file. Defaults to None, which will datetime.now() as unique name.
            mode (str, optional): Mode in which the h5 file is to be accessed. Defaults to "r". If the file doesn't exist and has to be created -> "a", otherwise "r" or "w".
            raw (np.ndarray, optional): Raw image data. Defaults to np.zeros(256, 256, 220).
            label (np.ndarray, optional): Labels for raw image. Defaults to np.zeros(256, 256, 220).

        Returns:
            f (h5py.File): Created hdf5 file.
        """
        assert target_path is not None, "No target path provided."

        if name is None:
            name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        target_file = os.path.join(target_path, name + ".h5")
        f = h5py.File(target_file, mode)
        if mode != "r":
            f.create_dataset("raw", data=raw)
            f.create_dataset("label", data=label)
        return f

    def get_file_name(self, file_path=None):
        # Some names are Axxx_L or Axxx_R. Those have to be included, otherwise names are onlz Axxx.
        file_name = os.path.basename(file_path)[:6]
        if file_name[4:] == "_L" or file_name[4:] == "_R" or file_name[4:] == "_M":
            return file_name
        return file_name[0:4]

    def nifty_to_numpy(self, raw_file=None, label_file=None):
        raw_data = nib.load(raw_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()
        return (raw_data, label_data)


def test_read(case=None):
    h5_file = os.path.join(os.getcwd(), "data", "h5", case + ".h5")
    f = h5py.File(h5_file, "r")
    raw = f["raw"][:]
    label = f["label"][:]
    return (raw, label)


def main():
    ### Get raw and mask images from (in this case) /Aneurysm-Detection/data/training
    filepath_training = os.path.join(os.getcwd(), "data", "training")
    filepaths_raw = sorted(glob.glob(os.path.join(filepath_training, "*_orig.*")))
    filepaths_masks = sorted(glob.glob(os.path.join(filepath_training, "*_masks.*")))

    ### Specify the target directory where the h5 files are to be stored and start the conversion.
    ### In this case: /Aneurysm-Detection/data/h5
    target_path = os.path.join(os.getcwd(), "data", "h5")
    converter = Nifty_Converter(filepaths_raw, filepaths_masks)
    converter.nifty_to_h5(target_path)

    ### Test if you can correctly read the data out of the hdf5 file.
    ### The test_read function reads the data the same way the 3DUnet implementation does.
    # raw, label = test_read(case="A003")
    # print(raw)

    logging.info("FINISHED")


if __name__ == "__main__":
    main()
