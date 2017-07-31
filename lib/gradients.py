import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class VanillaGrad(object):

    def __init__(self, pretrained_model, cuda=False):
        self.pretrained_model = pretrained_model
        self.features = pretrained_model.features
        self.cuda = cuda
        #self.pretrained_model.eval()

    def __call__(self, x, index=None):
        output = self.pretrained_model(x)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        if self.cuda:
            one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
        else:
            one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_variables=True)

        grad = x.grad.data.cpu().numpy()
        grad = grad[0, :, :, :]

        return grad


class SmoothGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=0.15,
                 n_samples=25, magnitude=True):
        super(SmoothGrad, self).__init__(pretrained_model, cuda)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitutde = magnitude

    def __call__(self, x, index=None):
        x = x.data.cpu().numpy()
        stdev = self.stdev_spread * (np.max(x) - np.min(x))
        total_gradients = np.zeros_like(x)
        for i in range(self.n_samples):
            noise = np.random.normal(0, stdev, x.shape).astype(np.float32)
            x_plus_noise = x + noise
            if self.cuda:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise).cuda(), requires_grad=True)
            else:
                x_plus_noise = Variable(torch.from_numpy(x_plus_noise), requires_grad=True)
            output = self.pretrained_model(x_plus_noise)

            if index is None:
                index = np.argmax(output.data.cpu().numpy())

            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][index] = 1
            if self.cuda:
                one_hot = Variable(torch.from_numpy(one_hot).cuda(), requires_grad=True)
            else:
                one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
            one_hot = torch.sum(one_hot * output)

            if x_plus_noise.grad is not None:
                x_plus_noise.grad.data.zero_()
            one_hot.backward(retain_variables=True)

            grad = x_plus_noise.grad.data.cpu().numpy()

            if self.magnitutde:
                total_gradients += (grad * grad)
            else:
                total_gradients += grad
            #if self.visdom:

        avg_gradients = total_gradients[0, :, :, :] / self.n_samples

        return avg_gradients


class GuidedBackpropReLU(torch.autograd.Function):

    def __init__(self, inplace=False):
        super(GuidedBackpropReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            input,
            pos_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors

        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, pos_mask_1),
                pos_mask_2)

        return grad_input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + inplace_str + ')'


class GuidedBackpropGrad(VanillaGrad):

    def __init__(self, pretrained_model, cuda=False):
        super(GuidedBackpropGrad, self).__init__(pretrained_model, cuda)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()


class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self, pretrained_model, cuda=False, stdev_spread=.15, n_samples=25, magnitude=True):
        super(GuidedBackpropSmoothGrad, self).__init__(
            pretrained_model, cuda, stdev_spread, n_samples, magnitude)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ is 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()


class FeatureExtractor(object):

    def __init__(self, model, target_layers):
        self.model = model
        self.features = model.features
        self.target_layers = target_layers
        self.gradients = []

    def __call__(self, x):
        target_activations, output = self.extract_features(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output

    def get_gradients(self):
        return self.gradients

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def extract_features(self, x):
        outputs = []
        for name, module in self.features._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class GradCam(object):

    def __init__(self, pretrained_model, target_layer_names, cuda):
        self.pretrained_model = pretrained_model
        self.cuda = cuda
        if self.cuda:
            self.pretrained_model.cuda()
        self.pretrained_model.eval()
        self.extractor = FeatureExtractor(self.pretrained_model, target_layer_names)

    def __call__(self, x, index=None):
        features, output = self.extractor(x)

        if index is None:
            index = np.argmax(output.data.cpu().numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = one_hot.cuda()
        one_hot = torch.sum(one_hot * output)

        self.pretrained_model.zero_grad()
        one_hot.backward(retain_variables=True)

        grads = self.extractor.get_gradients()[-1].data.cpu().numpy()

        target = features[-1].data.cpu().numpy()[0, :]
        weights = np.mean(grads, axis=(2, 3))[0, :]
        cam = np.ones(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam
