import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets

def gram(tensor):
    """
    Why the style transfer use gram matrix?
    Explained: https://www.zhihu.com/question/49805962
    """
    B, C, H, W = tensor.shape
    x = tensor.view(B, C, H * W)
    x_t = x.transpose(1, 2)
    return  torch.bmm(x, x_t) / (C * H * W)

def load_image(path):
    img = cv2.imread(path)  # BGR
    return img

# Show image
def show(img):
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # imshow() only accepts float [0,1] or int [0,255]
    img = np.array(img / 255).clip(0, 1)

    plt.figure(figsize = (10, 5))
    plt.imshow(img)
    plt.show()

def saveimg(img, image_path):
    img = img.clip(0, 255)
    cv2.imwrite(image_path, img)

def img2tensor(img):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    tensor = transform(img)

    # Unsqueeze for the batch_size dimension
    tensor = tensor.unsqueeze(dim = 0)
    return tensor

def tensor2img(tensor):
    # Remove the batch_size dimension
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()

    # Transpose from [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    return img

def save_loss_hist(c_loss, s_loss, total_loss, path, title = "Loss History"):
    x = [i for i in range(len(total_loss))]
    plt.figure(figsize = [10, 6])
    plt.plot(x, c_loss, label = "Content Loss")
    plt.plot(x, s_loss, label = "Style Loss")
    plt.plot(x, total_loss, label = "Total Loss")

    plt.legend(loc = 'best')
    plt.xlabel('Every 500 iterations')
    plt.ylabel('Loss')
    plt.title(title)
    plt.savefig(path)


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset for includes image file paths.
    Extends torchvision.datasets.ImageFolder()
    Reference: https://discuss.pytorch.org/t/dataloader-filenames-in-each-batch/4212/2
    """
    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)

        # the image file path
        path = self.imgs[index][0]

        # make a new tuple that includes original and the path
        tuple_with_path = (*original_tuple, path)
        return tuple_with_path