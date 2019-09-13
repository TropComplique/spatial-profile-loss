import torch
import copy
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from networks.generators import GlobalGenerator, LocalEnhancer
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


class pix2pix:

    def __init__(self, device, num_steps, with_enhancer=False, checkpoint=None):

        # in and out channels
        # for the generator:
        a, b = 1, 3

        if with_enhancer:
            G = LocalEnhancer(a, b, depth=64, downsample=3, num_blocks=9, enhancer_num_blocks=3)
            D = MultiScaleDiscriminator(a + b, depth=64, downsample=3, num_networks=3)
        else:
            G = GlobalGenerator(a, b, depth=64, downsample=3, num_blocks=9)
            D = MultiScaleDiscriminator(a + b, depth=64, downsample=3, num_networks=2)

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                init.ones_(m.weight)
                init.zeros_(m.bias)

        self.G = G.apply(weights_init).to(device)
        self.D = D.apply(weights_init).to(device)

        if with_enhancer:
            generator_state = torch.load(checkpoint + '_generator.pth')
            discriminator_state = torch.load(checkpoint + '_discriminator.pth')
            self.G.generator.load_state_dict(generator_state)
            self.D.load_state_dict(discriminator_state, strict=False)

        self.optimizer = {
            'G': optim.Adam(self.G.parameters(), lr=1e-3, betas=(0.5, 0.999)),
            'D': optim.Adam(self.D.parameters(), lr=1e-3, betas=(0.5, 0.999)),
        }

        def lambda_rule(i):
            decay = num_steps // 2
            m = 1.0 if i < decay else 1.0 - (i - decay) / decay
            return max(m, 0.0)

        self.schedulers = []
        for o in self.optimizer.values():
            self.schedulers.append(LambdaLR(o, lr_lambda=lambda_rule))

        self.gan_loss = LSGAN()
        self.feature_loss = FeatureLoss()

        # self.perceptual_loss = PerceptualLoss().to(device)
        self.perceptual_loss = nn.L1Loss()
        self.cp_loss = CPLoss()
        self.gp_loss = GPLoss()

        if USE_FLOAT16:
            models, optimizers = [self.D, self.G], [self.optimizer['D'], self.optimizer['G']]
            models, optimizers = amp.initialize(models, optimizers, opt_level='O2', num_losses=2)
            [self.D, self.G], [self.optimizer['D'], self.optimizer['G']] = models, optimizers

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

        self.optimizer['G'].zero_grad()
        # self.optimizer['D'].zero_grad()

        B_restored = self.G(A)
        # it has shape [n, b, h, w]

        # fake_scores, fake_features = self.D(B_restored, A)
        # fake_loss = self.gan_loss(fake_scores, False)

        # true_scores, true_features = self.D(B, A)
        # true_loss = self.gan_loss(true_scores, True)

        # feature_loss = self.feature_loss(fake_features, true_features)
        # gan_loss = self.gan_loss(fake_scores, True)
        cp_loss = self.cp_loss(B_restored, B)
        gp_loss = self.gp_loss(B_restored, B)
        reconstruction_loss = cp_loss + gp_loss
        # reconstruction_loss = self.perceptual_loss(B_restored, B)

        # discriminator_loss = 0.5 * (fake_loss + true_loss)
        generator_loss = reconstruction_loss# + 10 * feature_loss + gan_loss

        # self.D.requires_grad_(False)
        #self.G.requires_grad_(True)

        if not USE_FLOAT16:
            generator_loss.backward()#(retain_graph=True)
        else:
            with amp.scale_loss(generator_loss, self.optimizer['G'], loss_id=0) as loss_scaled:
                loss_scaled.backward()

        self.optimizer['G'].step()
        #self.D.requires_grad_(True)
        #self.G.requires_grad_(False)

#         if not USE_FLOAT16:
#             discriminator_loss.backward()
#         else:
#             with amp.scale_loss(discriminator_loss, self.optimizer['D'], loss_id=1) as loss_scaled:
#                 loss_scaled.backward()

#         self.optimizer['D'].step()
#         self.G.requires_grad_(True)

        # decay the learning rate
        for s in self.schedulers:
            s.step()

        # running average of weights
        accumulate(self.G_ema, self.G)

        loss_dict = {
            #'fake_loss': fake_loss.item(),
            #'true_loss': true_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'cp_loss': cp_loss.item(),
            'gp_loss': gp_loss.item(),
            #'feature_loss': feature_loss.item(),
           # 'gan_loss': gan_loss.item(),
            'generator_loss': generator_loss.item(),
            #'discriminators_loss': discriminator_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path + '_generator.pth')
        torch.save(self.G_ema.state_dict(), model_path + '_generator_ema.pth')
        #torch.save(self.D.state_dict(), model_path + '_discriminator.pth')
