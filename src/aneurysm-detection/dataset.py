from typing import Tuple
from pathlib import Path

import nibabel as nib

from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def load_dataloader(
    data_path: str = "data/training", batch_size: int = 4, shuffle: bool = True
) -> DataLoader:
    data_path = Path(data_path)

    # get case names
    cases = list(
        map(
            lambda file: "_".join(file.parts[-1].split("_")[:-1]),
            data_path.glob("*_orig.nii.gz"),
        )
    )

    assert cases, f"No cases found at {data_path}!"

    # create dataset + dataloader
    aneurysm_dataset = AeurysmDataset(data_path, cases)

    return DataLoader(
        aneurysm_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1
    )


class AeurysmDataset(Dataset):
    """Aneurysm dataset."""

    def __init__(
        self, data_path, cases, transform=None, image_size=(256, 256, 220)
    ):

        self.data_path = data_path
        self.cases = cases
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.cases)

    def __load_image(self, case):
        # load NIFTI image
        image_nib = nib.load(self.data_path / f"{case}_orig.nii.gz")
        return image_nib.get_fdata()

    def __load_mask(self, case):
        # load NIFTI mask
        mask_nib = nib.load(self.data_path / f"{case}_masks.nii.gz")
        return mask_nib.get_fdata()

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:

        # load image and ground-truth
        case = self.cases[idx]
        image = self.__load_image(case)[
            : self.image_size[0], : self.image_size[1], : self.image_size[2]
        ]
        mask = self.__load_mask(case)[
            : self.image_size[0], : self.image_size[1], : self.image_size[2]
        ]

        # data augmentation
        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)

        return sample["image"], sample["mask"]
