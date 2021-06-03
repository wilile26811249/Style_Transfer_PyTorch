import os
import argparse
import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import utils
import vgg16
import transform_net

os.makedirs('./weights', exist_ok = True)
os.makedirs('./style_weight', exist_ok = True)

parser = argparse.ArgumentParser(description = 'Style Transfer Project')
parser.add_argument('--epochs', type = int, default = 1,
                    help = 'Number of the training epochs')
parser.add_argument('--lr', type = float, default = 1e-3, metavar='LR',
                    help='Learning rate (default: 0.001)')
parser.add_argument('--batch-size', type = int, default = 4,
                    help = "Batch size of trainign and evaluation")
parser.add_argument('--img-size', type = int, default = 256,
                    help = "Training image size")
parser.add_argument('--content-weight', type = int, default = 5,
                    help = "Content weight for the final loss")
parser.add_argument('--style-weight', type = int, default = 100,
                    help = "Style weight for the final loss")
parser.add_argument('--save-interval', type = int, default = 100,
                    help = "Save model when every update save-interval times")
parser.add_argument('--content-dir', type = str, default ="./images/content_images",
                    help = 'Path for the content image root (default: ./images/content_images')
parser.add_argument('--style-img-path', type = str, default = "./images/style_images/mosaic.jpg",
                    help = 'Path for the style image path (default: ./images/style_images/mosaic.jpg')
parser.add_argument('--save-img-path', type = str, default = "./images/result_images",
                    help = 'Path for the content image root (default: ./images/result_images')
parser.add_argument('--save-model-path', type = str, default = './weights',
                    help = 'Path for the model weight (default: ./weights)')
parser.add_argument('--seed', type = int, default = 1,
                    help = 'Set the random seed (default: 1)')
parser.add_argument('--gpu-id', type = str, default = "0",
                    help = 'Select the sepcific GPU card (default: 0)')
parser.add_argument('--style-model-path', type = str, default = "style_transform",
                    help = 'Specific the final file name of the model weight (default: style_transform)')

def fix_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def train(args, device, VGG, TransformerNet, train_loader):
    # Optimizer
    optimizer = optim.Adam(TransformerNet.parameters(), lr = args.lr)
    # Loss Function
    MSELoss = nn.MSELoss().to(device)

    # Get Style Features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype = torch.float32).reshape(1, 3, 1, 1).to(device)
    style_image = utils.load_image(args.style_img_path)
    style_tensor = utils.img2tensor(style_image).to(device)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    B, C, H, W = style_tensor.shape
    style_features = VGG(style_tensor.expand([args.batch_size, C, H, W]))
    style_gram = {}
    for key, value in style_features.items():
        style_gram[key] = utils.gram(value)


    # Loss history
    content_loss_history = []
    style_loss_history = []
    total_loss_history = []
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0

    # Optimization/Training Loop
    train_steps = 1
    start_time = time.time()

    # Start Training
    for epoch in range(args.epochs):
        for content_batch, _ in train_loader:
            train_steps += 1
            # Get current batch size in case of odd batch sizes
            current_batch_size = content_batch.shape[0]

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Zero-out Gradients
            optimizer.zero_grad()

            # Generate images and get features
            content_batch = content_batch[:, [2, 1, 0]].to(device)
            generated_batch = TransformerNet(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))

            # Content Loss
            content_loss = args.content_weight * MSELoss(generated_features['relu2_2'], content_features['relu2_2'])
            batch_content_loss_sum += content_loss

            # Style Loss
            style_loss = 0.0
            for key, value in generated_features.items():
                temp_loss = MSELoss(utils.gram(value), style_gram[key][: current_batch_size])
                style_loss += temp_loss
            style_loss *= args.style_weight
            batch_style_loss_sum += style_loss.item()

            # Total Loss
            total_loss = content_loss + style_loss
            batch_total_loss_sum += total_loss.item()

            # Backprop and Weight Update
            total_loss.backward()
            optimizer.step()

            if (train_steps + 1) % args.save_interval == 0:
                # Print Losses
                print(f"========Iteration {train_steps}/{args.epochs * len(train_loader)}========")
                print(f"\tContent Loss:\t{batch_content_loss_sum / train_steps:.2f}")
                print(f"\tStyle Loss:\t{batch_style_loss_sum / train_steps:.2f}")
                print(f"\tTotal Loss:\t{batch_total_loss_sum / train_steps:.2f}")
                print(f"Time elapsed:\t{time.time() - start_time} seconds")

                # Save Model
                checkpoint_path = args.save_model_path + "checkpoint_" + str(train_steps) + ".pth"
                torch.save(TransformerNet.state_dict(), checkpoint_path)
                print(f"Saved TransformerNetwork checkpoint file at {checkpoint_path}")

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = utils.tensor2img(sample_tensor.clone().detach())
                sample_image_path = args.save_img_path + "sample_" + str(train_steps) + ".jpg"
                utils.saveimg(sample_image, sample_image_path)
                print(f"Saved sample tranformed image at {sample_image_path}")

                # Save loss histories
                content_loss_history.append(batch_total_loss_sum / train_steps)
                style_loss_history.append(batch_style_loss_sum / train_steps)
                total_loss_history.append(batch_total_loss_sum / train_steps)
    stop_time = time.time()
    print("Done Training the Transformer Network!")
    print(f"Training Time Costs: {stop_time - start_time} seconds")
    print("========Content Loss========")
    print(content_loss_history)
    print("========Style Loss========")
    print(style_loss_history)
    print("========Total Loss========")
    print(total_loss_history)

    # Save TransformerNetwork weights
    TransformerNet.eval()
    TransformerNet.cpu()
    final_path = os.path.join("./style_weights", args.style_model_path + ".pth")
    print(f"Saving TransformerNetwork weights at {final_path}")
    torch.save(TransformerNet.state_dict(), final_path)
    print("Done saving final model")

def main():
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    fix_seed(args.seed)

    # Generate Dataset and Dataloader
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(args.content_dir, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size = args.batch_size, shuffle = True)

    VGG = vgg16.vgg16_pretrained().to(device)
    TransformerNet = transform_net.TransformNet().to(device)
    train(args, device, VGG, TransformerNet, train_loader)

if __name__ == "__main__":
    main()