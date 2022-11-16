# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
from tqdm import tqdm

import torch
import todos
import video_basic

# NotImplementedError: The following operators are not implemented: ['aten::copy_']

SO_B, SO_C, SO_H, SO_W = 1, 3, 512, 512

def compile():
    model, device = video_basic.get_tvm_model()

    with torch.no_grad():
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        model = torch.jit.trace(model, input.to(device))

    pdb.set_trace()

    todos.data.mkdir("output")
    if not os.path.exists("output/video_zoom4x.so"):
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        with torch.no_grad():
            todos.tvmod.compile(model, device, input, "output/video_zoom4x.so")
    todos.model.reset_device()

def predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    device = todos.model.get_device()
    tvm_model = todos.tvmod.load("output/video_zoom4x.so", "cuda")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    mean_time = 0
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        orig_tensor = input_tensor.clone().detach()

        B, C, H, W = input_tensor.shape
        input_tensor = todos.data.resize_tensor(input_tensor, SO_H, SO_W)

        start_time = time.time()
        input_tensor = input_tensor.unsqueeze(0)
        predict_tensor = todos.tvmod.forward(tvm_model, input_tensor)
        predict_tensor = predict_tensor.squeeze(0)

        torch.cuda.synchronize()
        mean_time += time.time() - start_time

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        predict_tensor = todos.data.resize_tensor(predict_tensor, H, W)
        todos.data.save_tensor([orig_tensor, predict_tensor], output_file)

    mean_time = mean_time / len(image_filenames)

    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    todos.model.reset_device()


if __name__ == "__main__":
    compile()
    # predict("images/*.png", "output/so")
