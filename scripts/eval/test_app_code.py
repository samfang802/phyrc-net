#!/usr/bin/env python3
"""CPU unit test for the appearance code (docs/redesign/appearance_code.md).
  1. app_code_dim = 0: no appearance parameters.
  2. gate alpha_a = 0: the output ignores the code (any index, null, mean).
  3. gate != 0: the code changes the output; null (-1) and mean (-2) differ from a table code.
  4. eval mode is deterministic; train mode (dropout / noise) is not.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.model.adm import ADMUNetCFM

Z, S = 256, 4


def make(app):
    torch.manual_seed(0)
    net = ADMUNetCFM(
        in_channels=1, out_channels=1, model_channels=16, channel_mult=(1, 2, 2), num_res_blocks=1, cond_dim=32,
        latent_dim=3, use_split_conditioning=True, use_damage_mask_condition=True,
        physics_dim=Z + S + (1 if app else 0), damage_scalar_dim=S, use_vss_l3_u3=False,
        app_code_dim=8 if app else 0,
    )
    for n, p in net.named_parameters():  # AdaGN projections are zero-init; randomise so conditioning is visible
        if n not in ("alpha_a",) and p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.2)
    with torch.no_grad():
        net.alpha.fill_(0.5)
        net.alpha_d.fill_(0.5)
    return net.eval()


def run(net, idx, seed=1):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(4, 1, 16, 16, generator=g)
    t = torch.rand(4, generator=g)
    c = torch.randn(4, 3, generator=g)
    phys = torch.randn(4, Z + S, generator=g)
    if idx is not None:
        phys = torch.cat([phys, torch.tensor(idx, dtype=torch.float32).view(-1, 1)], dim=1)
    with torch.no_grad():
        return net(z, t, c, gru_feat=phys, damage_mask=torch.zeros(4, 1, 16, 16))


def main():
    base = make(False)
    assert not any(k.startswith(("app_", "e_a_", "alpha_a")) for k, _ in base.named_parameters())
    print("1 ok: no appearance parameters when off")
    net = make(True)
    assert net.alpha_a.abs().sum() == 0
    outs = [run(net, [i] * 4) for i in (0, 3, 7, -1, -2)]
    for o in outs[1:]:
        assert torch.equal(outs[0], o), "gate 0 must ignore the code"
    # same weights as the no-app net: the output equals the v11 forward
    print("2 ok: gate 0 ignores the code (max diff %.2e)" % max((outs[0] - o).abs().max().item() for o in outs))
    with torch.no_grad():
        net.alpha_a.fill_(0.5)
    o = {i: run(net, [i] * 4) for i in (0, 3, -1, -2)}
    for a, b in ((0, 3), (0, -1), (0, -2), (-1, -2)):
        assert (o[a] - o[b]).abs().max() > 1e-6, (a, b)
    print("3 ok: code, null and mean change the output when the gate is open")
    assert torch.equal(run(net, [3] * 4), run(net, [3] * 4))
    net.train(); net.app_drop_prob = 0.5; net.app_noise_std = 0.1; net.app_mix_prob = 0.5
    assert not torch.equal(run(net, [3] * 4), run(net, [3] * 4))
    print("4 ok: eval deterministic, train stochastic")


if __name__ == "__main__":
    main()
