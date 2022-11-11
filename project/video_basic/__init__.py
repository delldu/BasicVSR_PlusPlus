"""Video Basic Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch

import redos
import todos

from . import basic

import pdb

def get_tvm_model():
    """
    TVM model base on torch.jit.trace, much more orignal than torch.jit.script
    That's why we construct it from DeepGuidedFilterAdvanced
    """
    device = todos.model.get_device()
    model = basic.zoom_model()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def get_zoom_model():
    """Create model."""

    device = todos.model.get_device()
    model = basic.zoom_model()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    return model, device


def video_forward(model, input_tensor, device, scale=1, batch_size=10):
    B, C, H, W = input_tensor.size()  # (100, 3, 180, 320)

    output_tensor = torch.zeros(B, C, H * scale, W * scale)

    d_list = list(range(0, B, batch_size))
    progress_bar = tqdm(total=len(d_list))
    for i in d_list:
        progress_bar.update(1)

        input_tensor_clip = input_tensor[i : i + batch_size, :, :]
        out_clip = todos.model.forward(model, device, input_tensor_clip)
        output_tensor[i : i + batch_size, :, :, :] = out_clip

    return output_tensor.clamp(0.0, 1.0)


def get_zoom4x_model():
    """Create model."""

    device = todos.model.get_device()
    model = basic.VideoZoom4XModel()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)
    print(model.graph)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_zoom4x.torch"):
        model.save("output/video_zoom4x.torch")

    return model, device


def video_zoom4x_predict(input_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_zoom4x_model()

    print(f"  zoom4x {input_file}, save to {output_file} ...")
    lq_list = []

    def zoom_video_frame(no, data):
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)

    video.forward(callback=zoom_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device, scale=4)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()
    return True


def get_deblur_model():
    """Create model."""

    device = todos.model.get_device()

    model = basic.VideoDeblurModel()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_deblur.torch"):
        model.save("output/video_deblur.torch")

    return model, device


def video_deblur_predict(input_file, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_deblur_model()

    print(f"  deblur {input_file}, save to {output_file} ...")
    lq_list = []

    def deblur_video_frame(no, data):
        input_tensor = todos.data.frame_totensor(data)
        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        lq_list.append(input_tensor)

    video.forward(callback=deblur_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True


def get_denoise_model():
    """Create model."""

    device = todos.model.get_device()

    model = basic.VideoDenoiseModel()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")
    model = torch.jit.script(model)

    todos.data.mkdir("output")
    if not os.path.exists("output/video_denoise.torch"):
        model.save("output/video_denoise.torch")

    return model, device


def video_denoise_predict(input_file, sigma, output_file):
    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_denoise_model()

    print(f"  denoise {input_file}, save to {output_file} ...")
    lq_list = []

    def denoise_video_frame(no, data):
        input_tensor = todos.data.frame_totensor(data)
        # input_tensor[:, 3:4, :, :] = sigma / 255.0  # Add noise strength
        lq_list.append(input_tensor[:, 0:3, :, :])

    video.forward(callback=denoise_video_frame)

    lq = torch.cat(lq_list, dim=0)
    hq = video_forward(model, lq, device)

    for i in range(hq.shape[0]):
        # save image
        output_tensor = hq[i, :, :, :]
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        todos.data.save_tensor(output_tensor, temp_output_file)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i + 1)
        os.remove(temp_output_file)
    os.removedirs(output_dir)

    todos.model.reset_device()

    return True
