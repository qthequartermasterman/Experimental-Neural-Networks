import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch


def convert(cv_image):
    return cv_image[..., ::-1]


def sigmoid(x):
    # x is an nparray
    # This sigmoid will scale/translate in order to spread its input values over a wider range of outputs.
    mean = np.mean(x)
    variance = np.var(x)
    return 1 / (1 + np.exp(-1 * (x - mean) / variance ** 2))


def get_euclid_norm(x):
    # x is an nparray.
    # Takes the SobelX and SobelY gradients then finds the element-wise squareroot of the sum of the squares of those.
    # Also normalizes the values.
    sx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=7)
    sy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=7)
    euclid_norm = np.sqrt(np.square(sx) + np.square(sy))
    euclid_norm = euclid_norm / np.amax(euclid_norm)
    return euclid_norm


def generate_mask(x):
    # x is an nparray that is the input image in grayscale

    # use a gaussian blur to get rid of a lot of noise
    blur = cv2.GaussianBlur(x, (5, 5), 0)

    # kernel = np.ones((3, 3), np.uint8)  # the "brush" we will use in the morphologies.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    l2norm = get_euclid_norm(blur)
    l2norm = l2norm * 255 / np.amax(l2norm)
    l2norm = l2norm.astype('uint8')
    l2norm = cv2.dilate(l2norm, kernel)
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_OPEN, kernel)  # get ride of small blotches of white
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_GRADIENT, kernel)  # complete the lines
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_CLOSE, kernel)
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_ERODE, np.ones((5, 5), np.uint8))  # erode excess, to denoise
    l2norm = cv2.dilate(l2norm, kernel)
    l2norm = cv2.dilate(l2norm, kernel)
    l2norm = cv2.dilate(l2norm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    l2norm = cv2.dilate(l2norm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    l2norm = cv2.dilate(l2norm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # erode excess, to denoise
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # erode excess, to denoise
    l2norm = cv2.morphologyEx(l2norm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # erode excess, to denoise
    # print('l2norm', l2norm)
    ret, thresh = cv2.threshold(l2norm, 5, 255, 0)
    # print('threshold', thresh)
    im2 = thresh
    # Blur the edges of our mask to compensate for errors
    im2 = cv2.GaussianBlur(im2, (7, 7), 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours', contours)



    for contour in contours:
        im2 = cv2.drawContours(im2, [contour], 0, 255, -1)

    color_filled = cv2.cvtColor(im2.astype('uint8'), cv2.COLOR_GRAY2RGB)  # convert mask to a 3d image
    color_filled = color_filled.astype('float') / 255
    color_filled = .6 * color_filled + .4  # we want to always be able to see the entire image, so we scale the mask
    return im2, color_filled


def mask_image(x, concatenate=True):
    # x is an nparray representing an RGB image.
    # concatenate determines whether or not we should concatenate the image mask as an extra channel. This effectively
    # makes it an RGBA image. Defaults to False and will as such simply multiply the original image by its mask.
    grayscale = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    im2, color_imgmask = generate_mask(grayscale)
    if concatenate is False:
        return x.astype('float') * color_imgmask
    else:
        # normalized_mask = im2.astype('float')/255
        normalized_mask = np.atleast_3d(im2)  # Make this a channeled image
        return np.concatenate([x, normalized_mask], axis=2)


def tensor_to_image(tensor):
    # print("converting tensor to image")
    imgpy = tensor.numpy()
    imgpy = np.swapaxes(imgpy, 0, 2)
    # print("image from tensor", imgpy)
    return imgpy


def image_to_tensor(image):
    img = np.swapaxes(image, 0, 2)
    return torch.from_numpy(img)


def denormalize_image(x_original):
    x = x_original
    x = x / 2 + 0.5
    x = x * 255 / np.amax(x)
    return x.astype('uint8')


def renormalize_image(x_original):
    x = x_original
    x = x.astype('float32')
    x = x / np.amax(x)
    x = 2 * (x - .5)
    return x


def mask_normalized_image(x_original):
    x = x_original
    x = denormalize_image(x)
    x = mask_image(x)
    x = renormalize_image(x)
    return x


def mask_pytorch_image(x_original):
    x = x_original
    x = tensor_to_image(x)
    x = mask_normalized_image(x)
    return image_to_tensor(x)


def mask_pytorch_image_batch(batch):
    new_batch_list = []
    for i in range(batch.shape[0]):
        new_batch_list.append(mask_pytorch_image(batch[i]))
    return torch.stack(new_batch_list)


def test_mask():
    # img = cv2.imread('GettyImages-547031277-58ef97803df78cd3fc724e24.jpg',0)
    img_color = convert(cv2.imread('../geico_1300_v5.jpg', cv2.CV_64F))
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

    imgmask, color_imgmask = generate_mask(img)
    # masked_image = img_color.astype('float') * color_imgmask
    masked_image = mask_image(img_color)

    plt.subplot(2, 2, 1), plt.imshow(img_color, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    # plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.xticks([]), plt.yticks([])
    plt.title('Euclidean Norm'), plt.imshow(get_euclid_norm(img), cmap='gray')
    plt.subplot(2, 2, 3), plt.imshow(masked_image.astype('uint8'), cmap='gray')
    plt.title('Masked image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    # plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(imgmask, cmap='gray')
    plt.title('Image Mask'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    # plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

    plt.show()
    return masked_image
