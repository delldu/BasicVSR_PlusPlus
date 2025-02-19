# Copyright (c) OpenMMLab. All rights reserved.
import tempfile

import mmcv
import pytest
import torch
from mmcv.runner import obj_from_dict

from mmedit.models import build_model
from mmedit.models.backbones import TDANNet
from mmedit.models.losses import MSELoss


def test_tdan_model():

    model_cfg = dict(
        type="TDAN",
        generator=dict(
            type="TDANNet",
            in_channels=3,
            mid_channels=64,
            out_channels=3,
            num_blocks_before_align=5,
            num_blocks_after_align=10,
        ),
        pixel_loss=dict(type="MSELoss", loss_weight=1.0, reduction="sum"),
        lq_pixel_loss=dict(type="MSELoss", loss_weight=1.0, reduction="sum"),
    )

    train_cfg = None
    test_cfg = None

    # build restorer
    restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg)

    # test attributes
    assert restorer.__class__.__name__ == "TDAN"
    assert isinstance(restorer.generator, TDANNet)
    assert isinstance(restorer.pixel_loss, MSELoss)

    # prepare data
    inputs = torch.rand(1, 5, 3, 8, 8)
    targets = torch.rand(1, 3, 32, 32)

    # test train_step and forward_test (gpu)
    if torch.cuda.is_available():
        restorer = restorer.cuda()
        data_batch = {"lq": inputs.cuda(), "gt": targets.cuda()}

        # prepare optimizer
        optim_cfg = dict(type="Adam", lr=2e-4, betas=(0.9, 0.999))
        optimizer = {
            "generator": obj_from_dict(optim_cfg, torch.optim, dict(params=getattr(restorer, "generator").parameters()))
        }

        # train_step
        outputs = restorer.train_step(data_batch, optimizer)
        assert isinstance(outputs, dict)
        assert isinstance(outputs["log_vars"], dict)
        assert isinstance(outputs["log_vars"]["loss_pix"], float)
        assert outputs["num_samples"] == 1
        assert torch.equal(outputs["results"]["lq"], data_batch["lq"].cpu())
        assert torch.equal(outputs["results"]["gt"], data_batch["gt"].cpu())
        assert torch.is_tensor(outputs["results"]["output"])
        assert outputs["results"]["output"].size() == (1, 3, 32, 32)

        # test forward_dummy
        with torch.no_grad():
            output = restorer.forward_dummy(data_batch["lq"])
        assert isinstance(output, tuple)
        assert torch.is_tensor(output[0])
        assert output[0].size() == (1, 3, 32, 32)
        assert torch.is_tensor(output[1])
        assert output[1].size() == (1, 5, 3, 8, 8)

        # forward_test
        with torch.no_grad():
            outputs = restorer(**data_batch, test_mode=True)
        assert torch.equal(outputs["lq"], data_batch["lq"].cpu())
        assert torch.equal(outputs["gt"], data_batch["gt"].cpu())
        assert torch.is_tensor(outputs["output"])
        assert outputs["output"].size() == (1, 3, 32, 32)

        with torch.no_grad():
            outputs = restorer(inputs.cuda(), test_mode=True)
        assert torch.equal(outputs["lq"], data_batch["lq"].cpu())
        assert torch.is_tensor(outputs["output"])
        assert outputs["output"].size() == (1, 3, 32, 32)

    # test with metric and save image
    if torch.cuda.is_available():
        train_cfg = mmcv.ConfigDict(tsa_iter=1)
        test_cfg = dict(metrics=("PSNR", "SSIM"), crop_border=0)
        test_cfg = mmcv.Config(test_cfg)

        data_batch = {
            "lq": inputs.cuda(),
            "gt": targets.cuda(),
            "meta": [{"gt_path": "fake_path/fake_name.png", "key": "000/00000000"}],
        }

        restorer = build_model(model_cfg, train_cfg=train_cfg, test_cfg=test_cfg).cuda()

        with pytest.raises(AssertionError):
            # evaluation with metrics must have gt images
            restorer(lq=inputs.cuda(), test_mode=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = restorer(**data_batch, test_mode=True, save_image=True, save_path=tmpdir, iteration=None)
            assert isinstance(outputs, dict)
            assert isinstance(outputs["eval_result"], dict)
            assert isinstance(outputs["eval_result"]["PSNR"], float)
            assert isinstance(outputs["eval_result"]["SSIM"], float)

            outputs = restorer(**data_batch, test_mode=True, save_image=True, save_path=tmpdir, iteration=100)
            assert isinstance(outputs, dict)
            assert isinstance(outputs["eval_result"], dict)
            assert isinstance(outputs["eval_result"]["PSNR"], float)
            assert isinstance(outputs["eval_result"]["SSIM"], float)

            with pytest.raises(ValueError):
                # iteration should be number or None
                restorer(**data_batch, test_mode=True, save_image=True, save_path=tmpdir, iteration="100")
