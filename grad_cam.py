import argparse
import os
import sys

import numpy as np
from scipy import misc
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg16, vgg19
from torchvision.utils import save_image

from lib.gradients import GradCam, GuidedBackpropGrad
from lib.image_utils import preprocess_image, save_cam_image, save_as_gray_image
from lib.labels import IMAGENET_LABELS


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--img', type=str, default='',
	                    help='Input image path')
	parser.add_argument('--out_dir', type=str, default='./result/cam/',
	                    help='Result directory path')
	args = parser.parse_args()
	args.cuda = args.cuda and torch.cuda.is_available()
	if args.cuda:
	    print("Using GPU for acceleration")
	else:
	    print("Using CPU for computation")
	if args.img:
		print('Input image: {}'.format(args.img))
	else:
		print('Input image: raccoon face (scipy.misc.face())')
	print('Output directory: {}'.format(args.out_dir))
	print()
	return args


def main():
	args = parse_args()

	if not os.path.exists(args.out_dir):
		os.makedirs(args.out_dir)

	target_layer_names = ['35']
	target_index = None

	# Prepare input image
	if args.img:
		img = cv2.imread(args.img, 1)
	else:
		img = misc.face()
	img = np.float32(cv2.resize(img, (224, 224))) / 255
	preprocessed_img = preprocess_image(img, args.cuda)

	# Prediction
	output = vgg19(pretrained=True)(preprocessed_img)
	pred_index = np.argmax(output.data.cpu().numpy())
	print('Prediction: {}'.format(IMAGENET_LABELS[pred_index]))

	# Prepare grad cam
	grad_cam = GradCam(
		pretrained_model=vgg19(pretrained=True),
		target_layer_names=target_layer_names,
		cuda=args.cuda)

		# Compute grad cam
	mask = grad_cam(preprocessed_img, target_index)

	save_cam_image(img, mask, os.path.join(args.out_dir, 'grad_cam.jpg'))
	print('Saved Grad-CAM image')

	# Reload preprocessed image
	preprocessed_img = preprocess_image(img)

	# Compute guided backpropagation
	guided_backprop = GuidedBackpropGrad(
		pretrained_model=vgg19(pretrained=True), cuda=args.cuda)
	guided_backprop_saliency = guided_backprop(preprocessed_img, index=target_index)

	cam_mask = np.zeros(guided_backprop_saliency.shape)
	for i in range(guided_backprop_saliency.shape[0]):
		cam_mask[i, :, :] = mask

	cam_guided_backprop = np.multiply(cam_mask, guided_backprop_saliency)
	save_as_gray_image(
		cam_guided_backprop,
		os.path.join(args.out_dir, 'guided_grad_cam.jpg'))
	print('Saved Guided Grad-CAM image')


if __name__ == '__main__':
    main()
