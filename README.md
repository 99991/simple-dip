# simple-dip
Simplified version of https://github.com/DmitryUlyanov/deep-image-prior

* So far, only inpainting has been implemented.
* No guarantees for exactly identical implementations.

## How to

1. Install PyTorch, NumPy, Pillow and OpenCV (or uncomment the three lines with `cv2` if you do not want the preview while training)
2. Run `inpaint.py`

This will download the library test image and mask from https://github.com/DmitryUlyanov/deep-image-prior and inpaint the masked region.

## Example

| Masked image | Inpainted image |
|:------------:|:-------:|
|<img width="500" alt="Masked library" src="https://user-images.githubusercontent.com/18725165/181269409-ec77715c-81e4-4026-8875-a8911cb868ac.png">|<img width="500" alt="Inpainted library" src="https://user-images.githubusercontent.com/18725165/181269420-ee57895d-fc7c-4861-b115-b446fb26a3d9.png">|


