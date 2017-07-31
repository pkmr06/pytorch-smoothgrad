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

from lib.gradients import VanillaGrad, SmoothGrad, GuidedBackpropGrad, GuidedBackpropSmoothGrad
from lib.image_utils import preprocess_image, save_as_gray_image
from lib.labels import IMAGENET_LABELS


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda', action='store_true', default=False,
	                    help='Use NVIDIA GPU acceleration')
	parser.add_argument('--img', type=str, default='',
	                    help='Input image path')
	parser.add_argument('--out_dir', type=str, default='./result/grad/',
	                    help='Result directory path')
	parser.add_argument('--n_samples', type=int, default=10,
	                    help='Sample size of SmoothGrad')
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
	print('Sample size of SmoothGrad: {}'.format(args.n_samples))
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

	# Compute vanilla gradient
	vanilla_grad = VanillaGrad(
		pretrained_model=vgg19(pretrained=True), cuda=args.cuda)
	vanilla_saliency = vanilla_grad(preprocessed_img, index=target_index)
	save_as_gray_image(vanilla_saliency, os.path.join(args.out_dir, 'vanilla_grad.jpg'))
	print('Saved vanilla gradient image')

	# Reload preprocessed image
	preprocessed_img = preprocess_image(img, args.cuda)

	# Compute guided gradient
	guided_grad = GuidedBackpropGrad(
		pretrained_model=vgg19(pretrained=True), cuda=args.cuda)
	guided_saliency = guided_grad(preprocessed_img, index=target_index)
	save_as_gray_image(guided_saliency, os.path.join(args.out_dir, 'guided_grad.jpg'))
	print('Saved guided backprop gradient image')

	# Reload preprocessed image
	preprocessed_img = preprocess_image(img, args.cuda)

	# Compute smooth gradient
	smooth_grad = SmoothGrad(
		pretrained_model=vgg19(pretrained=True),
		cuda=args.cuda,
		n_samples=args.n_samples,
		magnitude=True)
	smooth_saliency = smooth_grad(preprocessed_img, index=target_index)
	save_as_gray_image(smooth_saliency, os.path.join(args.out_dir, 'smooth_grad.jpg'))
	print('Saved smooth gradient image')

	# Reload preprocessed image
	preprocessed_img = preprocess_image(img, args.cuda)

	# Compute guided smooth gradient
	guided_smooth_grad = GuidedBackpropSmoothGrad(
		pretrained_model=vgg19(pretrained=True),
		cuda=args.cuda,
		n_samples=args.n_samples,
		magnitude=True)
	guided_smooth_saliency = guided_smooth_grad(preprocessed_img, index=target_index)
	save_as_gray_image(guided_smooth_saliency, os.path.join(args.out_dir, 'guided_smooth_grad.jpg'))
	print('Saved guided backprop smooth gradient image')


if __name__ == '__main__':
    main()
