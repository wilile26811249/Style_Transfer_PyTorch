# Fast Neural Style Transfer implement by PyTorch!
**Author: Willie Chen**

This repository contains a ```PyTorch``` implementation of an algorithm for fast style transfer. The algorithm can be used to mix the content of the input image with the style of the style image.

---

## Table of Contents
* [Network structure overview](#network-structure-overview)
* [Image Stylization Result](#image-stylization-result)
* [Usage](#usage)
	- [Training the Style-Transfer Network](#training-the-style-transfer-network)
	- [Neural Style Transfer](#neural-style-transfer)
* [Reference](#reference)
* [Acknowledgement](#acknowledgement)

---

# Network structure overview

**Network architecture detail: [LINK](#reference)**
<p align=center> <img src ="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-319-46475-6_43/MediaObjects/419974_1_En_43_Fig2_HTML.gif" width="420px" border="1"> </p>
<p align=center><b>Network Overview</b></p>

<p>


## Image Stylization Result
---
<p align=center> <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/input_images/sunset.jpg" width="420px" border="1"> </p>
<p align=center><b>Original Image (Source: https://www.pixiv.net/artworks/75323963)</b></p>

<table style="width:100%, border:3px">
  <tr>
    <th style="text-align:center">Style Image</th>
    <th style="text-align:center">Stylize image of the original image</th>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/candy.jpg"
        height="252px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/sunset_candy.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/mosaic.jpg"
        height="252"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/sunset_mosaic.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/starry-night.jpg"
        height="252px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/sunset_starry_night.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/great_wave.jpg"
        height="252px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/sunset_great_wave.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/guernica.jpg"
        height="252px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/sunset_guernica.jpg"   border="1">
    </td>
  </tr>
</table>

---

# **Usage**


# Training the Style-Transfer Network

0. Download the coco dataset
``` bash
bash download_dataset.sh
```
1. Train the Style Transfer model

```train.py```: Train the Transform Network that learn the style from the ```style_image``` and retain the semantic-information about the ```input_image```.
```bash
python neural_style/train.py --content-dir ./images/content_images --style-img-path ./images/style_images/mosaic.jpg --epochs 1 --batch-size 4
```
**Arguments (Optional)**
```
usage: train_success.py [-h] [--epochs EPOCHS] [--lr LR]
                        [--batch-size BATCH_SIZE] [--img-size IMG_SIZE]
                        [--content-weight CONTENT_WEIGHT]
                        [--style-weight STYLE_WEIGHT]
                        [--save-interval SAVE_INTERVAL]
                        [--content-dir CONTENT_DIR]
                        [--style-img-path STYLE_IMG_PATH]
                        [--save-img-path SAVE_IMG_PATH]
                        [--save-model-path SAVE_MODEL_PATH] [--seed SEED]
                        [--gpu-id GPU_ID]
                        [--style-model-path STYLE_MODEL_PATH]

Style Transfer Project

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of the training epochs
  --lr LR               Learning rate (default: 0.001)
  --batch-size BATCH_SIZE
                        Batch size of trainign and evaluation
  --img-size IMG_SIZE   Training image size
  --content-weight CONTENT_WEIGHT
                        Content weight for the final loss
  --style-weight STYLE_WEIGHT
                        Style weight for the final loss
  --save-interval SAVE_INTERVAL
                        Save model when every update save-interval times
  --content-dir CONTENT_DIR
                        Path for the content image root (default:
                        ./images/content_images
  --style-img-path STYLE_IMG_PATH
                        Path for the style image path (default:
                        ./images/style_images/starry-night-cropped.jpg
  --save-img-path SAVE_IMG_PATH
                        Path for the content image root (default:
                        ./images/result_images
  --save-model-path SAVE_MODEL_PATH
                        Path for the model weight (default: ../weights)
  --seed SEED           Set the random seed (default: 1)
  --gpu-id GPU_ID       Select the sepcific GPU card (default: 0)
  --style-model-path STYLE_MODEL_PATH
                        Specific the final file name of the model weight
                        (default: style_transform)
```
---

# Neural Style Transfer

0. Download the pretrained weight

Pretrained Weight: https://drive.google.com/drive/folders/1Iy-JGUA-KFjY0OgRmzhl2HQmaaXrHYjh?usp=sharing

1. Run the fast style transfer

```bash
python neural_style/stylize_inference.py
```


**Hint:**
* images/input_images: The picture you want to transfer style, please put it in this folder

---

## Reference
1. **J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for
real-time style transfer and super-resolution. ECCV 2016**
2. **J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual Losses for Real-Time Style Transfer and Super-Resolution: Supplementary Material**

---

## Acknowledgement

The code benefits from outstanding prior work and their implementations including:
- [Perceptual losses for real-time style transfer and super-resolution](https://arxiv.org/pdf/1603.08155.pdf) by Johnson *et al. 2016* and its torch implementation [code](https://github.com/jcjohnson/neural-style) by Johnson.
