# The scipt is modified from: 
#   1. https://github.com/AlexiaJM/RelativisticGAN/tree/master
#   2. https://github.com/CompVis/latent-diffusion/tree/main

import torch, functools
import torch.nn as nn

# Initialize weights
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		# Estimated variance, must be around 1
		m.weight.data.normal_(1.0, 0.02)
		# Estimated mean, must be around 0
		m.bias.data.fill_(0)


# DCGAN generator
class DCGAN_G(torch.nn.Module):
    def __init__(self, image_size, z_dim, condition_dim, SELU, G_h_size):
        super(DCGAN_G, self).__init__()
        
        main = torch.nn.Sequential()

        # We need to know how many layers we will use at the beginning
        mult = image_size // 8

        ### Start block
        # Z_size random numbers
        
        main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(z_dim + condition_dim, G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        if SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(G_h_size * mult))
            main.add_module('Start-ReLU', torch.nn.ReLU())
        # Size = (G_h_size * mult) x 4 x 4

        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(G_h_size * mult, G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            
            if SELU:
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(G_h_size * (mult//2)))
                main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU())
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1

        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(G_h_size, 3, kernel_size=4, stride=2, padding=1, bias=False))
        main.add_module('End-Tanh', torch.nn.Tanh())
        # Size = n_colors x image_size x image_size
        self.main = main

    def forward(self, input, condition):
        input = torch.cat((input, condition), 1)
        output = self.main(input)
        return output
    
# DCGAN discriminator (using somewhat the reverse of the generator)
class DCGAN_D(torch.nn.Module):
    def __init__(self, image_size, image_dim, condition_dim, spectral_D, D_h_size, SELU, no_batch_norm):
        super(DCGAN_D, self).__init__()
        self.image_size = image_size
        self.image_dim = image_dim
        self.dense = torch.nn.Linear(condition_dim, 1*image_size*image_size)
        main = torch.nn.Sequential()

        ### Start block
        # Size = n_colors x image_size x image_size
        if spectral_D:
            main.add_module('Start-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(image_dim + 1, D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        else:
            main.add_module('Start-Conv2d', torch.nn.Conv2d(image_dim + 1, D_h_size, kernel_size=4, stride=2, padding=1, bias=False))
        if SELU:
            main.add_module('Start-SELU', torch.nn.SELU(inplace=True))
        else:
            main.add_module('Start-LeakyReLU', torch.nn.LeakyReLU(0.2, inplace=True))
        image_size_new = image_size // 2
        # Size = D_h_size x image_size/2 x image_size/2

        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            if spectral_D:
                main.add_module('Middle-SpectralConv2d [%d]' % i, torch.nn.utils.spectral_norm(torch.nn.Conv2d(D_h_size * mult, D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            else:
                main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(D_h_size * mult, D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False))
            if SELU:
                main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            else:
                if not spectral_D and not no_batch_norm:
                    main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(D_h_size * (2*mult)))
                main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        if spectral_D:
            main.add_module('End-SpectralConv2d', torch.nn.utils.spectral_norm(torch.nn.Conv2d(D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))
        else:
            main.add_module('End-Conv2d', torch.nn.Conv2d(D_h_size * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))
        # if loss_D in [1]:
        #     main.add_module('End-Sigmoid', torch.nn.Sigmoid())
        # Size = 1 x 1 x 1 (Is a real cat or not?)
        self.main = main

    def forward(self, input, condition):
        condition = condition.view(-1, condition.size(1))
        condition = self.dense(condition).view(-1, 1, self.image_size, self.image_size)
        input = torch.cat((input, condition), 1)
        output = self.main(input)
        # Convert from 1 x 1 x 1 to 1 so that we can compare to given label (cat or not?)
        return output.view(-1)






class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h

if __name__ == "__main__":
    batch = 4
    image_size = 128
    noise_length = 128
    feature_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    noise = torch.randn(batch, noise_length, 1, 1).to(device)
    condition = torch.randn(batch, feature_size, 1, 1).to(device)
    real = torch.randn(batch, 3, image_size, image_size).to(device)
    fake = torch.randn(batch, 3, image_size, image_size).to(device)
    
    # Generator
    G = DCGAN_G(image_size=image_size, z_dim=noise_length, condition_dim=feature_size, SELU=False, G_h_size=128).to(device)
    D = DCGAN_D(image_dim=3, image_size=image_size, condition_dim=feature_size, SELU=True, spectral_D=False, D_h_size=128, no_batch_norm=True).to(device)
    print(G)
    print(D)
    
    
    # fake_image = G(noise, condition)
    # print(fake_image)
    # critic = D(fake_image, condition)
    # print(critic)
    
    
    
    
    