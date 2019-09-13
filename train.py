import os
import numpy as np
import cv2
import skimage
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from input_pipeline import Images
from model import Model

from torch.backends import cudnn
cudnn.benchmark = True


MODELS_DIR = 'models/'
LOGS_DIR = 'summaries/'

BATCH_SIZE = 10  # float16 training
DATA = '/home/dan/datasets/feidegger/images/'
NUM_EPOCHS = 50
SIZE = (480, 320)  # height and width
METHOD = 'easy'  # 'easy' or 'hard'

DEVICE = torch.device('cuda:0')
MODEL_NAME = 'run00'

SAVE_EPOCH = 10
PLOT_IMAGE_STEP = 500
PLOT_LOSS_STEP = 10

IS_SECOND_STAGE = False
if IS_SECOND_STAGE:
    CHECKPOINT = os.path.join(MODELS_DIR, MODEL_NAME + f'_epoch_{NUM_EPOCHS}')
    MODEL_NAME += '_second_stage'
    NUM_EPOCHS = 30
    SIZE = (2 * SIZE[0], 2 * SIZE[1])


class Simplifier:

    def __init__(self):
        pass

    def simplify(self, x):
        """
        Arguments:
            x: a float tensor with shape [3, h, w].
        Returns:
            a float tensor with shape [1, h, w].
        """
        x = x.numpy()
        x = 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]
        x *= 255.0
        x = skimage.filters.gaussian(x, sigma=1)
        x = skimage.filters.scharr(x)
        q1 = np.quantile(x, 0.5)
        q2 = np.quantile(x, 0.95)
        x[x < q1] = 0
        q2 = max(1e-8, q2)
        x = x/q2
        x = np.clip(x, 0.0, 1.0)
        x = torch.FloatTensor(x).unsqueeze(0)
        return x

    def __call__(self, B):
        """
        The main algorithm will try
        to restore B from A.

        Arguments:
            B: a float tensor with shape [n, b, h, w].
        Returns:
            A: a float tensor with shape [n, a, h, w].
        """
        A = []
        for x in B:
            A.append(self.simplify(x))
        return torch.stack(A)


def main():

    log_dir = os.path.join(LOGS_DIR, MODEL_NAME)
    writer = SummaryWriter(log_dir)

    dataset = Images(folder=DATA, size=SIZE, method=METHOD)
    # dataset = FewImages(DATA, size=512, num_samples=1000)

    data_loader = DataLoader(
        dataset=dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    num_steps = NUM_EPOCHS * (len(dataset) // BATCH_SIZE)

    model = pix2pix(
        device=DEVICE, num_steps=num_steps,
        with_enhancer=IS_SECOND_STAGE,
        checkpoint=CHECKPOINT if IS_SECOND_STAGE else None
    )
    simplifier = Simplifier()

    representations = []
    indices = np.random.randint(0, len(dataset), size=10)

    for k, i in enumerate(indices):

        B = dataset[i].unsqueeze(0)
        A = simplifier(B)

        writer.add_image(f'sample_{k}', B[0], 0)
        writer.add_image(f'sample_{k}', A[0], 1)

        representations.append(A.to(DEVICE))

    # number of weight updates
    i = 0

    for e in range(1, NUM_EPOCHS + 1):
        for images in data_loader:

            B = images
            A = simplifier(B).to(DEVICE)
            B = B.to(DEVICE)

            # some images are almost empty
            is_A_bad = A.var(dim=[1, 2, 3]).min().item() < 1e-3
            is_B_bad = B.var(dim=[1, 2, 3]).min().item() < 1e-3
            if is_A_bad or is_B_bad:
                continue

            i += 1
            start = time.perf_counter()
            losses = model.train_step(A, B)
            step_time = time.perf_counter() - start
            step_time = round(1000 * step_time, 1)

            if i % PLOT_IMAGE_STEP == 0:

                for j, A in enumerate(representations):
                    with torch.no_grad():
                        B = model.G(A)
                    writer.add_image(f'sample_{j}', B[0].cpu(), i)

            if i % PLOT_LOSS_STEP == 0:

                for k, v in losses.items():
                    writer.add_scalar(f'losses/{k}', v, i)

            print(f'epoch {e}, iteration {i}, time {step_time} ms')

        if e % SAVE_EPOCH == 0:
            save_path = os.path.join(MODELS_DIR, MODEL_NAME + f'_epoch_{e}')
            model.save_model(save_path)


if __name__ == '__main__':
    main()
