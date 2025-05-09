from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt


#   Dataset format:
# LABEL [-1,1] [-1,1] [-1,1] ...\nLABEL [-1,1] ...
#   i.e (with arbitrary/made-up values):
# 4.0000 -1.0000 -0.9480 -0.5610 ...
# 7.0000 -0.7490 0.1370 -0.3710 ...
# 0.0000 1.0000 -1.0000 -0.8719 ...
# 1.0000 0.1648 1.0000 0.6302 ...
# 5.0000 0.0737 -0.9006 0.2750 ...
# 7.0000 -0.7910 0.2175 -0.3443 ...
#   etc... (or just look at `dataset/test-data.txt` or `dataset/train-data.txt`)
def vectorize_dataset(path_to_dataset):
    dataset = np.loadtxt(path_to_dataset)  # LABEL [-1,1] [-1,1] [-1,1] ...\nLABEL [-1,1] ...
    labels  = dataset[:, 0].astype(int)  # LABEL\n LABEL\n LABEL\n ...
    pixels  = dataset[:, 1:]  # [-1,1] [-1,1] [-1,1] ...

    return pixels, labels


def image_to_grayscale_vector(path_to_image: str, display_processed_image: bool=False) -> np.ndarray:
    image   = convert_image_to_grayscale(path_to_image)
    image   = invert_colors(image)
    image   = crop_to_image_content(image)
    image   = pad_square(image)
    image   = resize_16(image)
    array   = scale_array(image)
    if display_processed_image:
        show_processed_image(array)

    return array.flatten()


def convert_image_to_grayscale(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def invert_colors(img: Image.Image) -> Image.Image:
    return ImageOps.invert(img)


def crop_to_image_content(img: Image.Image, thresh: int = 20) -> Image.Image:
    a       = np.array(img)
    ys, xs  = np.where(a > thresh)
    if ys.size and xs.size:
        top, bottom = ys.min(), ys.max() + 1
        left, right = xs.min(), xs.max() + 1
        return img.crop((left, top, right, bottom))

    return img


def pad_square(image: Image.Image, fill: int = 0) -> Image.Image:
    width, height   = image.size
    largest_side    = max(width, height)
    canvas  = Image.new("L", (largest_side, largest_side), color=fill)
    canvas.paste(image, ((largest_side - width)//2, (largest_side - height)//2))

    return canvas


def resize_16(img: Image.Image) -> Image.Image:
    return img.resize((16, 16), Image.Resampling.LANCZOS)


def scale_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)/127.5 - 1.0


def show_processed_image(arr: np.ndarray) -> None:
    plt.title("Processed Image")
    plt.imshow(arr, cmap="gray", vmin=-1, vmax=1)
    plt.axis("off")
    plt.show()
