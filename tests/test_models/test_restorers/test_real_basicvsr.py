# Copyright (c) OpenMMLab. All rights reserved.
from unittest.mock import patch

import pytest
import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import RealBasicVSRNet
from mmedit.models.components import UNetDiscriminatorWithSpectralNorm
from mmedit.models.losses import GANLoss, L1Loss


def test_real_basicvsr():

    model_cfg = dict(
        type="RealBasicVSR",
        generator=dict(type="RealBasicVSRNet"),
        discriminator=dict(
            type="UNetDiscriminatorWithSpectralNorm", in_channels=3, mid_channels=64, skip_connection=True
        ),
        pixel_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
        cleaning_loss=dict(type="L1Loss", loss_weight=1.0, reduction="mean"),
        gan_loss=dict(type="GANLoss", gan_type="vanilla", loss_weight=1e-1, real_label_val=1.0, fake_label_val=0),
        is_use_sharpened_gt_in_pixel=True,
        is_use_sharpened_gt_in_percep=True,
        is_use_sharpened_gt_in_gan=True,
        is_use_ema=True,
    )

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == "RealBasicVSR"
    assert isinstance(restorer.generator, RealBasicVSRNet)
    assert isinstance(restorer.discriminator, UNetDiscriminatorWithSpectralNorm)
    assert isinstance(restorer.pixel_loss, L1Loss)
    assert isinstance(restorer.gan_loss, GANLoss)

    # prepare data
    inputs = torch.rand(1, 5, 3, 64, 64)
    targets = torch.rand(1, 5, 3, 256, 256)
    data_batch = {"lq": inputs, "gt": targets, "gt_unsharp": targets}

    # prepare optimizer
    optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.9, 0.999))
    optimizer = {
        "generator": obj_from_dict(optim_cfg, torch.optim, dict(params=getattr(restorer, "generator").parameters())),
        "discriminator": obj_from_dict(
            optim_cfg, torch.optim, dict(params=getattr(restorer, "discriminator").parameters())
        ),
    }

    # no forward train in GAN models, raise ValueError
    with pytest.raises(ValueError):
        restorer(**data_batch, test_mode=False)

    # test train_step
    with patch.object(restorer, "perceptual_loss", return_value=(torch.tensor(1.0), torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        for v in ["loss_perceptual", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix", "loss_clean"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (5, 3, 256, 256)

    # test train_step (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        optimizer = {
            "generator": obj_from_dict(
                optim_cfg, torch.optim, dict(params=getattr(restorer, "generator").parameters())
            ),
            "discriminator": obj_from_dict(
                optim_cfg, torch.optim, dict(params=getattr(restorer, "discriminator").parameters())
            ),
        }
        data_batch = {"lq": inputs.cuda(), "gt": targets.cuda(), "gt_unsharp": targets.cuda()}

        # train_step
        with patch.object(
            restorer, "perceptual_loss", return_value=(torch.tensor(1.0).cuda(), torch.tensor(2.0).cuda())
        ):
            outputs = restorer.train_step(data_batch, optimizer)
            assert isinstance(outputs, dict)
            assert isinstance(outputs["log_vars"], dict)
            for v in ["loss_perceptual", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix", "loss_clean"]:
                assert isinstance(outputs["log_vars"][v], float)
            assert outputs["num_samples"] == 1
            assert torch.equal(outputs["results"]["lq"], data_batch["lq"].cpu())
            assert torch.equal(outputs["results"]["gt"], data_batch["gt"].cpu())
            assert torch.is_tensor(outputs["results"]["output"])
            assert outputs["results"]["output"].size() == (5, 3, 256, 256)

    # test disc_steps and disc_init_steps and start_iter
    data_batch = {"lq": inputs.cpu(), "gt": targets.cpu(), "gt_unsharp": targets.cpu()}
    train_cfg = dict(disc_steps=2, disc_init_steps=2, start_iter=0)
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
    with patch.object(restorer, "perceptual_loss", return_value=(torch.tensor(1.0), torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        for v in ["loss_d_real", "loss_d_fake"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (5, 3, 256, 256)

    # test without pixel loss and perceptual loss
    model_cfg_ = model_cfg.copy()
    model_cfg_.pop("pixel_loss")
    restorer = build_model(model_cfg_, train_cfg=None, test_cfg=None)

    outputs = restorer.train_step(data_batch, optimizer)
    assert isinstance(outputs, dict)
    assert isinstance(outputs["log_vars"], dict)
    for v in ["loss_gan", "loss_d_real", "loss_d_fake"]:
        assert isinstance(outputs["log_vars"][v], float)
    assert outputs["num_samples"] == 1
    assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
    assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
    assert torch.is_tensor(outputs["results"]["output"])
    assert outputs["results"]["output"].size() == (5, 3, 256, 256)

    # test train_step w/o loss_percep
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)
    with patch.object(restorer, "perceptual_loss", return_value=(None, torch.tensor(2.0))):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        for v in ["loss_style", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix", "loss_clean"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (5, 3, 256, 256)

    # test train_step w/o loss_style
    restorer = build_model(model_cfg, train_cfg=None, test_cfg=None)
    with patch.object(restorer, "perceptual_loss", return_value=(torch.tensor(2.0), None)):
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        for v in ["loss_perceptual", "loss_gan", "loss_d_real", "loss_d_fake", "loss_pix", "loss_clean"]:
            assert isinstance(outputs["log_vars"][v], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(outputs["results"]["lq"], data_batch["lq"])
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"])
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (5, 3, 256, 256)
