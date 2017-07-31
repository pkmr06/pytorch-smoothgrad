import cv2
import numpy as np
import torch
from torch.autograd import Variable


def preprocess_image(img, cuda=False):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    if cuda:
        preprocessed_img = Variable(preprocessed_img.cuda(), requires_grad=True)
    else:
        preprocessed_img = Variable(preprocessed_img, requires_grad=True)

    return preprocessed_img


def save_as_gray_image(img, filename, percentile=99):
    img_2d = np.sum(img, axis=0)
    span = abs(np.percentile(img_2d, percentile))
    vmin = -span
    vmax = span
    img_2d = np.clip((img_2d - vmin) / (vmax - vmin), -1, 1)
    cv2.imwrite(filename, img_2d * 255)

    return


def save_cam_image(img, mask, filename):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(filename, np.uint8(255 * cam))
