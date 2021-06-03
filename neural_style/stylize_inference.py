import enum
import cv2
import glob
import os
import time

import torch
from torchvision import transforms

import utils
import transform_net


# Check the style weight path
style_filepath = {}
style_name_list = []
weight_path = glob.glob("../style_weight/*.pth")
number_of_style = len(weight_path)
for path in weight_path:
    weight_name = path.split("/")[-1]
    style_name = weight_name.replace(".pth", "")
    style_name_list.append(style_name)
    style_filepath[style_name] = weight_name

print(style_filepath)
device = "cuda" if torch.cuda.is_available() else "cpu"
inference_device = "GPU" if device == "cuda" else "CPU"
print(f'Now you use "{inference_device}" to processing Style Transfer')

net = transform_net.TransformNet()
with torch.no_grad():
    while True:
        print(f"You have {number_of_style} style can choice!")
        for index, (style_name, _) in enumerate(style_filepath.items()):
            print(f"({index}) {style_name}", end = "  ")

        style_choice = input(f"\nSelect the style you want (Input 0 ~ {number_of_style - 1}): ")
        if style_choice.isnumeric():
            style_choice = int(style_choice)
            if style_choice < 0 or style_choice > (number_of_style - 1):
                print(f"Range is from 0 to {number_of_style - 1}")
                break

            content_img_path = input("Enter the image path (Image should place in the input_images folder): ")
            dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'images/input_images'))
            content_img_path = os.path.join(dir_path, content_img_path)
            if not os.path.exists(content_img_path):
                print(f"Can't found img in {content_img_path}")
                break
            content_image = utils.load_image(content_img_path)
            starttime = time.time()
            content_tensor = utils.img2tensor(content_image).to(device)

            STYLE_TRANSFORM_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../style_weight', style_filepath[style_name_list[style_choice]]))
            net.load_state_dict(torch.load(STYLE_TRANSFORM_PATH))
            net = net.to(device)

            # Conver image to new style
            content_tensor = utils.img2tensor(content_image).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.tensor2img(generated_tensor.detach())
            print("Transfer Time: {}".format(time.time() - starttime))
            utils.show(generated_image)
        else:
            print(f"Please input (Input 1 ~ {number_of_style + 1}) to select style!")
