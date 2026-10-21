"""DCGAN generator / discriminator of the HystereticCGAN baseline (cGANckpt/), inference only.

Restored from the code removed in 9284bc7 (src/model/Component/generative.py, ImageGeneration.py
@ 965cbfa). Module names are unchanged so the Lightning checkpoint's `G.*` / `D.*` weights load as is.
Condition = [GRU hidden state (1024) ; direction one-hot (2)] (latent_conditioning, direction_conditioning).
"""

from __future__ import annotations

import torch


class DCGAN_G(torch.nn.Module):
    def __init__(self, image_size, z_dim, condition_dim, SELU, G_h_size):
        super().__init__()
        main = torch.nn.Sequential()
        mult = image_size // 8
        main.add_module("Start-ConvTranspose2d", torch.nn.ConvTranspose2d(z_dim + condition_dim, G_h_size * mult, 4, 1, 0, bias=False))
        if SELU:
            main.add_module("Start-SELU", torch.nn.SELU(inplace=True))
        else:
            main.add_module("Start-BatchNorm2d", torch.nn.BatchNorm2d(G_h_size * mult))
            main.add_module("Start-ReLU", torch.nn.ReLU())
        i = 1
        while mult > 1:
            main.add_module("Middle-ConvTranspose2d [%d]" % i, torch.nn.ConvTranspose2d(G_h_size * mult, G_h_size * (mult // 2), 4, 2, 1, bias=False))
            if SELU:
                main.add_module("Middle-SELU [%d]" % i, torch.nn.SELU(inplace=True))
            else:
                main.add_module("Middle-BatchNorm2d [%d]" % i, torch.nn.BatchNorm2d(G_h_size * (mult // 2)))
                main.add_module("Middle-ReLU [%d]" % i, torch.nn.ReLU())
            mult = mult // 2
            i += 1
        main.add_module("End-ConvTranspose2d", torch.nn.ConvTranspose2d(G_h_size, 3, 4, 2, 1, bias=False))
        main.add_module("End-Tanh", torch.nn.Tanh())
        self.main = main

    def forward(self, z, condition):
        return self.main(torch.cat((z, condition), 1))


class DCGAN_D(torch.nn.Module):
    def __init__(self, image_size, image_dim, condition_dim, spectral_D, D_h_size, SELU, no_batch_norm):
        super().__init__()
        self.image_size = image_size
        self.dense = torch.nn.Linear(condition_dim, image_size * image_size)
        sn = torch.nn.utils.spectral_norm if spectral_D else (lambda m: m)
        main = torch.nn.Sequential()
        main.add_module("Start-SpectralConv2d" if spectral_D else "Start-Conv2d", sn(torch.nn.Conv2d(image_dim + 1, D_h_size, 4, 2, 1, bias=False)))
        main.add_module("Start-SELU" if SELU else "Start-LeakyReLU", torch.nn.SELU(inplace=True) if SELU else torch.nn.LeakyReLU(0.2, inplace=True))
        size, mult, i = image_size // 2, 1, 0
        while size > 4:
            name = "Middle-SpectralConv2d [%d]" if spectral_D else "Middle-Conv2d [%d]"
            main.add_module(name % i, sn(torch.nn.Conv2d(D_h_size * mult, D_h_size * 2 * mult, 4, 2, 1, bias=False)))
            if SELU:
                main.add_module("Middle-SELU [%d]" % i, torch.nn.SELU(inplace=True))
            else:
                if not spectral_D and not no_batch_norm:
                    main.add_module("Middle-BatchNorm2d [%d]" % i, torch.nn.BatchNorm2d(D_h_size * 2 * mult))
                main.add_module("Middle-LeakyReLU [%d]" % i, torch.nn.LeakyReLU(0.2, inplace=True))
            size, mult, i = size // 2, mult * 2, i + 1
        main.add_module("End-SpectralConv2d" if spectral_D else "End-Conv2d", sn(torch.nn.Conv2d(D_h_size * mult, 1, 4, 1, 0, bias=False)))
        self.main = main

    def forward(self, x, condition):
        c = self.dense(condition.view(-1, condition.size(1))).view(-1, 1, self.image_size, self.image_size)
        return self.main(torch.cat((x, c), 1)).view(-1)


def load_cgan(ckpt_path, device="cuda"):
    """→ (G, D, hparams) from a HystereticCGAN Lightning checkpoint, both in eval mode."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = payload["hyper_parameters"]
    G = DCGAN_G(hp["image_size"], hp["z_dim"], hp["condition_dim"], hp["SELU"], hp["G_h_size"])
    D = DCGAN_D(hp["image_size"], 3, hp["condition_dim"], hp["spectral_D"], hp["D_h_size"], hp["SELU"], hp["no_batch_norm"])
    sd = payload["state_dict"]
    G.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("G.")})
    D.load_state_dict({k[2:]: v for k, v in sd.items() if k.startswith("D.")})
    keep = ("image_size", "z_dim", "condition_dim", "latent_conditioning", "direction_conditioning", "extractor_path", "data_path", "loss_type")
    meta = {k: hp.get(k) for k in keep}
    meta["epoch"] = payload.get("epoch")
    return G.to(device).eval(), D.to(device).eval(), meta
