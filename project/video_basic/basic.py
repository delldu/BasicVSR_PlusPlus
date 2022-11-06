"""Video Basic Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 10月 05日 星期三 01:31:30 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from typing import List
from typing import Dict
import pdb


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks."""
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN."""

    def __init__(self, mid_channels=64, res_scale=1.0):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front."""

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super(ResidualBlocksWithInputConv, self).__init__()
        main = []
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class SPyNet(nn.Module):
    """SPyNet network structure.

    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.

    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super(SPyNet, self).__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()  # [9, 3, 192, 320]

        # normalize the input images
        ref: List[torch.Tensor] = [(ref - self.mean) / self.std]  # ==> List
        supp: List[torch.Tensor] = [(supp - self.mean) / self.std]  # ==> List

        # generate downsampled frames
        for level in range(5):
            ref.append(F.avg_pool2d(ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(supp[-1], kernel_size=2, stride=2, count_include_pad=False))

        ref = ref[::-1]  # reverse list
        supp = supp[::-1]  # reverse list

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level, m in enumerate(self.basic_module):  # range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = (
                    F.interpolate(
                        flow, scale_factor=2.0, mode="bilinear", align_corners=True, recompute_scale_factor=True
                    )
                    * 2.0
                )

            # add the residue to the upsampled flow
            t = flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + m(torch.cat([ref[level], t, flow_up], dim=1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.

        This function computes the optical flow from ref to supp.

        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).

        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """

        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, relu=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
        )
        if relu:
            self.activate = nn.ReLU(inplace=True)
        else:
            self.activate = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activate(x)
        return x


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """

    def __init__(self):
        super(SPyNetBasicModule, self).__init__()
        self.basic_module = nn.Sequential(
            ConvModule(
                in_channels=8,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            ConvModule(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            ConvModule(
                in_channels=64,
                out_channels=32,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            ConvModule(
                in_channels=32,
                out_channels=16,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
            ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, relu=False),
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


def flow_warp(x, flow, interpolation: str = "bilinear", padding_mode: str = "zeros", align_corners: bool = True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f"The spatial sizes of input ({x.size()[-2:]}) and " f"flow ({flow.size()[1:3]}) are not the same."
        )
    _, _, h, w = x.size()
    # create mesh grid
    # grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing="ij")
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
    # grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    output = F.grid_sample(x, grid_flow, mode=interpolation, padding_mode=padding_mode, align_corners=align_corners)
    return output


class PixelShufflePack(nn.Module):
    """Pixel Shuffle upsample layer."""

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super(PixelShufflePack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2,
        )

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ModulatedDeformConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        deform_groups=1,
        bias=True,
    ):
        super(ModulatedDeformConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = 0

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, offset, mask):
        pass
        return x


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        # (Pdb) self.conv_offset
        # Sequential(
        #   (0): Conv2d(196, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (1): LeakyReLU(negative_slope=0.1, inplace=True)
        #   (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (3): LeakyReLU(negative_slope=0.1, inplace=True)
        #   (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #   (5): LeakyReLU(negative_slope=0.1, inplace=True)
        #   (6): Conv2d(64, 432, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # )

    def forward(self, x, extra_feat, flow_1, flow_2):
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)  # out.size() -- [1, 432, 135, 240]

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)  # [1, 288, 180, 320]

        # mask
        mask = torch.sigmoid(mask)  # [1, 144, 180, 320]

        y =  torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask,
        )

        return y



class BasicVSRPlusPlus(nn.Module):
    """BasicVSR++ network structure.
    Paper:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation
        and Alignment

    Args:
        mid_channels (int, optional): Channel number of the intermediate
            features. Default: 64.
        num_blocks (int, optional): The number of residual blocks in each
            propagation branch. Default: 7.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        is_low_res_input (bool, optional): Whether the input is low-resolution
            or not. If False, the output resolution is equal to the input
            resolution. Default: True.
    """

    def __init__(
        self,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
    ):
        super(BasicVSRPlusPlus, self).__init__()
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input

        self.spynet = SPyNet()
        if is_low_res_input:
            self.feat_extract = ResidualBlocksWithInputConv(3, mid_channels, 5)
        else:
            self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5),
            )

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            self.backbone[module] = ResidualBlocksWithInputConv((2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResidualBlocksWithInputConv(5 * mid_channels, mid_channels, 5)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2, upsample_kernel=3)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4.0, mode="bilinear", align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lqs) -> List[torch.Tensor]:
        """Compute optical flow using SPyNet for feature alignment.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        return flows_forward, flows_backward

    def propagate(self, feats: Dict[str, List[torch.Tensor]], flows, module_name: str):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
        """
        # Propagate map -----
        # backward_1, ['spatial', 'backward_1']
        # forward_1, ['spatial', 'backward_1', 'forward_1']
        # backward_2, ['spatial', 'backward_1', 'forward_1', 'backward_2']
        # forward_2, ['spatial', 'backward_1', 'forward_1', 'backward_2', 'forward_2']

        n, t, _, h, w = flows.size()  # [1, 11, 2, 135, 240]

        forward_frame_idx: List[int] = list(range(0, t + 1))  # [0, 1, ..., 11]
        forward_flow_idx: List[int] = list(range(-1, t))  # [-1, 0, ..., 10]

        backward_frame_idx: List[int] = list(range(t, -1, -1))  # [11, 10, ..., 0]
        backward_flow_idx: List[int] = backward_frame_idx

        pg_keys = ["backward_1", "forward_1", "backward_2", "forward_2"]
        feat_prop = flows.new_zeros(n, self.mid_channels, h, w).to(flows.device)

        if "forward" in module_name:
            frame_idx = forward_frame_idx
            flow_idx = forward_flow_idx
        else:
            frame_idx = backward_frame_idx
            flow_idx = backward_flow_idx

        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][idx]

            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)

                # feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
                for k, m in self.deform_align.items():
                    if k == module_name:
                        feat_prop = m(feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            # feat = [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]
            feat = [feat_current]
            for k in pg_keys:
                if k != module_name and k in feats.keys():
                    feat.append(feats[k][idx])
            feat.append(feat_prop)
            feat = torch.cat(feat, dim=1)

            # feat_prop = feat_prop + self.backbone[module_name](feat)
            for k, m in self.backbone.items():
                if k == module_name:
                    feat_prop = feat_prop + m(feat)

            feats[module_name].append(feat_prop)

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats: Dict[str, List[torch.Tensor]]):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h | h, 4w |w).
        """
        outputs = []
        for i in range(0, lqs.size(1)):
            # hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            # hr.insert(0, feats["spatial"][i])
            # hr = torch.cat(hr, dim=1)
            hr_list = []
            # feats.keys() -- ['spatial', 'backward_1', 'forward_1', 'backward_2', 'forward_2']
            for key in feats.keys():
                hr_list.append(feats[key][i])
            hr = torch.cat(hr_list, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:  # Zoom: True
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            outputs.append(hr)
        # outputs[i].size() -- [1, 3, 540, 960] for i in [0, 11]
        return torch.stack(outputs, dim=1)  # [1, 12, 3, 540, 960]

    def forward_x(self, lqs):
        """Forward function for BasicVSR++.
        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """
        n, t, c, h, w = lqs.size()  # [1, 21, 3, 180, 320]

        if self.is_low_res_input:  # Zoom: True
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w),
                scale_factor=0.25,
                mode="bicubic",
                recompute_scale_factor=True,
                align_corners=False,
            ).view(n, t, c, h // 4, w // 4)

        feats: Dict[str, List[torch.Tensor]] = {}
        # compute spatial features
        feats["spatial"] = []
        for i in range(0, t):
            feat = self.feat_extract(lqs[:, i, :, :, :])
            feats["spatial"].append(feat)
        #  feats.keys() -- ['spatial'], len(feats['spatial']) -- 12

        # compute optical flow using the low-res inputs
        # assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
        #     "The height and width of low-res inputs must be at least 64, " f"but got {h} and {w}."
        # )
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        # len(flows_forward), flows_forward[0].size()
        # (1, [12, 2, 180, 320])
        # len(flows_backward), flows_backward[0].size()
        # (1, [12, 2, 180, 320])

        # feature propgation
        pg_keys = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for module_name in pg_keys:
            feats[module_name] = []
            if "backward" in module_name:
                flows = flows_backward
            elif flows_forward is not None:
                flows = flows_forward
            else:
                flows = flows_backward.flip(1)

            self.propagate(feats, flows, module_name)

        return self.upsample(lqs, feats)

    def forward(self, x):
        # Define max GPU/CPU memory -- 4G
        max_h = 1024
        max_W = 1024
        multi_times = 8

        # Need Resize ?
        B, C, H, W = x.size()
        if H > max_h or W > max_W:
            s = min(max_h / H, max_W / W)
            SH, SW = int(s * H), int(s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x = x

        # Need Pad ?
        PH, PW = resize_x.size(2), resize_x.size(3)
        if PH % multi_times != 0 or PW % multi_times != 0:
            r_pad = multi_times - (PW % multi_times)
            b_pad = multi_times - (PH % multi_times)
            resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            resize_pad_x = resize_x

        y = self.forward_x(resize_pad_x.unsqueeze(0)).squeeze(0)
        del resize_pad_x, resize_x  # Release memory !!!

        if self.is_low_res_input:  # Zoom !!!
            y = y[:, :, 0 : 4 * PH, 0 : 4 * PW]  # Remove Pads
            y = F.interpolate(y, size=(4 * H, 4 * W), mode="bilinear", align_corners=False)
        else:
            # denoise/deblur
            y = y[:, :, 0:PH, 0:PW]  # Remove Pads
            y = F.interpolate(y, size=(H, W), mode="bilinear", align_corners=False)

        return y.clamp(0.0, 1.0)


def zoom_model():
    return BasicVSRPlusPlus(num_blocks=7, is_low_res_input=True)


def deblur_model():
    return BasicVSRPlusPlus(num_blocks=15, is_low_res_input=False)


def denoise_model():
    return BasicVSRPlusPlus(num_blocks=15, is_low_res_input=False)
