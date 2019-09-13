import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from generator import Generator
from sp_loss import CPLoss, GPLoss


USE_FLOAT16 = True
if USE_FLOAT16:
    from apex import amp


def accumulate(model_accumulator, model, decay=0.99):
    """Exponential moving average."""

    params = dict(model.named_parameters())
    ema_params = dict(model_accumulator.named_parameters())

    for k in params.keys():
        ema_params[k].data.mul_(decay).add_(1.0 - decay, params[k].data)


class Model:

    def __init__(self, device, num_steps):

        # in and out channels
        # for the generator:
        a, b = 1, 3

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

        G = Generator(a, b, depth=64, downsample=3, num_blocks=9)
        self.G = G.apply(weights_init).to(device)

        def lambda_rule(i):
            decay = num_steps // 10
            m = 1.0 if i < decay else 1.0 - (i - decay) / (num_steps - decay)
            return max(m, 0.0)

        self.optimizer = optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.5, 0.999))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_rule)

        self.cp_loss = CPLoss()
        self.gp_loss = GPLoss()

        if USE_FLOAT16:
            self.G, self.optimizer = amp.initialize(self.G, self.optimizer, opt_level='O2')

        # a copy for exponential moving average
        self.G_ema = copy.deepcopy(self.G)

    def train_step(self, A, B):
        """
        The input tensors represent images
        with pixel values in [0, 1] range.

        Arguments:
            A: a float tensor with shape [n, a, h, w].
            B: a float tensor with shape [n, b, h, w].
        Returns:
            a dict with float numbers.
        """
        self.optimizer.zero_grad()

        B_restored = self.G(A)
        # it has shape [n, b, h, w]

        cp_loss = self.cp_loss(B_restored, B)
        gp_loss = self.gp_loss(B_restored, B)
        reconstruction_loss = cp_loss + gp_loss

        if not USE_FLOAT16:
            reconstruction_loss.backward()
        else:
            with amp.scale_loss(generator_loss, self.optimizer['G'], loss_id=0) as loss_scaled:
                loss_scaled.backward()

        self.optimizer.step()
        self.scheduler.step()

        # running average of weights
        accumulate(self.G_ema, self.G)

        loss_dict = {
            'total_loss': reconstruction_loss.item(),
            'cp_loss': cp_loss.item(),
            'gp_loss': gp_loss.item()
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.G_ema.state_dict(), model_path + '_generator_ema.pth')
