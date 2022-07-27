import numpy as np
import torch
import torch.nn as nn
import os, cv2, urllib.request
from PIL import Image

def conv(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        kernel_size//2, padding_mode="reflect")

def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True))

def make_neural_network():
    # U-net WITHOUT skip connections. Interesting observation: Replacing the
    # encoder with random noise produces only slightly worse results.
    return nn.Sequential(
        conv_bn_relu( 1, 16, kernel_size=5, stride=2),
        conv_bn_relu(16, 16, kernel_size=5),
        
        conv_bn_relu(16, 32, kernel_size=5, stride=2),
        conv_bn_relu(32, 32, kernel_size=5),
        
        conv_bn_relu(32, 64, kernel_size=5, stride=2),
        conv_bn_relu(64, 64, kernel_size=5),
        
        conv_bn_relu( 64, 128, kernel_size=5, stride=2),
        conv_bn_relu(128, 128, kernel_size=5),
        
        conv_bn_relu(128, 128, kernel_size=5, stride=2),
        conv_bn_relu(128, 128, kernel_size=5),
        
        conv_bn_relu(128, 128, kernel_size=5, stride=2),
        conv_bn_relu(128, 128, kernel_size=5),
        
        nn.Upsample(scale_factor=2), conv_bn_relu(128, 128),
        nn.Upsample(scale_factor=2), conv_bn_relu(128, 128),
        nn.Upsample(scale_factor=2), conv_bn_relu(128, 128),
        nn.Upsample(scale_factor=2), conv_bn_relu(128,  64),
        nn.Upsample(scale_factor=2), conv_bn_relu( 64,  32),
        nn.Upsample(scale_factor=2), conv_bn_relu( 32,  16),
        
        conv(16, 3, kernel_size=1),
        
        nn.Sigmoid())

def download(url):
    filename = url.split("/")[-1]

    if not os.path.isfile(filename):
        print("Downloading", url)
        urllib.request.urlretrieve(url, filename)

    return filename

def main():
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_url = "https://raw.githubusercontent.com/DmitryUlyanov/deep-image-prior/master/data/inpainting/library.png"
    mask_url = "https://raw.githubusercontent.com/DmitryUlyanov/deep-image-prior/master/data/inpainting/library_mask.png"

    # Download and open input images
    image = Image.open(download(image_url)).convert("RGB")
    mask  = Image.open(download( mask_url)).convert("L")

    # Image size must be multiples of 64
    assert image.size[0] % 64 == 0
    assert image.size[0] % 64 == 0
    assert mask.size == image.size

    w, h = image.size

    # Convert PIL image to numpy float32 numpy array
    image = np.array(image).astype(np.float32) / 255.0
    mask  = np.array( mask).astype(np.float32) / 255.0

    # Convert [h, w, c] format to [n, c, h, w] image format preferred by PyTorch
    image = image[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
    mask  =  mask[np.newaxis, np.newaxis, :, :]

    # Convert NumPy arrays to PyTorch tensors on GPU
    image = torch.tensor(image, device=device)
    mask  = torch.tensor( mask, device=device)

    # Random noise as input for neural network
    noise = 0.1 * torch.rand((1, 1, h, w), device=device)

    # Assemble neural network
    net = make_neural_network().to(device)

    # Create optimizer
    optim = torch.optim.Adam(net.parameters(), lr=0.01)

    for iteration in range(3001):
        # Generate image from noise with neural network
        inpainted_image = net(noise)

        # Loss is difference of images where mask is white (black is ignored)
        difference = mask * (inpainted_image - image)
        loss = torch.mean(torch.square(difference))

        # Optimize loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(f"Iteration {iteration:05d} - mean squared error {loss.item():8.6f}")

        if iteration % 100 == 0:
            # Convert PyTorch image to NumPy image
            preview = inpainted_image[0].detach().cpu().numpy()
            preview = preview.transpose(1, 2, 0)
            preview = np.clip(preview * 255, 0, 255).astype(np.uint8)

            # Preview inpainted image
            cv2.imshow("preview", preview[:, :, ::-1])
            cv2.waitKey(1)

            # Save image
            Image.fromarray(preview).save(f"inpaint_{iteration:05d}.png")

if __name__ == "__main__":
    main()
 
