from modulefinder import Module
import numpy as np

from dataset import DataLoader, load_dataloader
from model import init_model


def train(model, dataloader: DataLoader):
    import torch
    from torch.optim import Adam
    from torch.nn import BCELoss

    # create Adam optimizer with standard parameters
    optimizer = Adam(params=model.parameters())

    # binary cross entropy loss
    loss_fx = BCELoss()

    epochs = 10
    losses = []

    for _ in range(epochs):

        # iterate dataloader
        epoch_loss = []
        for image, mask in dataloader:
            _, h, w, _ = image.shape
            image_slices = (
                image.permute(0, 3, 1, 2).reshape(-1, h, w).float().cuda()
            )
            image_slices = torch.unsqueeze(image_slices, axis=1)
            mask_slices = (
                mask.permute(0, 3, 1, 2).reshape(-1, h, w).float().cuda()
            )

            # reset gradient
            optimizer.zero_grad()

            # pass input through model
            output = model(image_slices)
            output = torch.squeeze(output)

            # calculate loss
            loss = loss_fx(output, mask_slices)
            epoch_loss.append(loss.detach().item())

            # back-propagation
            loss.backward()
            optimizer.step()

        #
        losses.append(np.mean(epoch_loss))


def main():
    model, dataloader = init_model(), load_dataloader(
        data_path="Aneurysm-Detection/data/training"
    )

    train(model, dataloader)


if __name__ == "__main__":
    main()
