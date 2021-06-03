# Style_Transfer_PyTorch

This repository is implement the fast-neural-style-tranfer.


Paper:

[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)<p>
Network architecture detail: [LINK](https://web.eecs.umich.edu/~justincj/papers/eccv16/JohnsonECCV16Supplementary.pdf)

<img src ="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-3-319-46475-6_43/MediaObjects/419974_1_En_43_Fig2_HTML.gif" width="420px" border="1" style="display:block; margin:auto;">
<div style="text-align:center"><strong>Network Overview</strong></div>

<p>


## **Image Stylization**
---
<img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/input_images/vivy.jpg" width="420px" border="1" style="display:block; margin:auto;">
<div style="text-align:center">Original Image</div>
<table style="width:100%, border:3px">
  <tr>
    <th style="text-align:center">Style Image</th>
    <th style="text-align:center">Stylize image of the original image</th>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/candy.jpg"
        height="210px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/vivy_candy.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/mosaic.jpg"
        height="210px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/vivy_mosaic.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/starry-night.jpg"
        height="210px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/vivy_starry_night.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/great_wave.jpg"
        height="210px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/vivy_great_wave.jpg"   border="1">
    </td>
  </tr>
  <tr>
    <td width=50% align="center">
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/style_images/guernica.jpg"
        height="210px"  border="1">
    </td>
    <td>
        <img src ="https://raw.githubusercontent.com/wilile26811249/Style_Transfer_PyTorch/main/images/result_images/vivy_guernica.jpg"   border="1">
    </td>
  </tr>
</table>

---