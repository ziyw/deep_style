from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import PIL
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

# Content Loss 
class ContentLoss(nn.Module):
    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion.forward(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

# Style loss 
class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d) 
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram.forward(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion.forward(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss



class StyleModel(object):
  
  def __init__(self, content_layers, style_layers, style_num=1, content_weight=1, style_weight=1000):
    
    super(StyleModel, self).__init__()
    
    self.content_layers = content_layers
    self.style_layers = style_layers 
    
    self.style_num = style_num
    self.content_weight = content_weight 
    self.style_weight = style_weight
    

  def transfer(self, content, styles):
    # Load images 
    self.content = self.load_image(content).type(torch.FloatTensor)
    self.style = self.load_image(style).type(torch.FloatTensor)

    if self.style_num == 1:
        self.single_combine()
        self.show_single_combine()

    else:
        self.multi_combine()





  def load_image(self,filename,size=200):
    loader = transforms.Compose([
    transforms.Resize((size,size)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(filename)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

  def unload_tensor(self, tensor,size=200):
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.view(3, size, size)  # remove the fake batch dimension
    image = unloader(image)
    return image
  
  def show_single_combine(self):
    fig = plt.figure()
    
    plt.subplot(131)
    plt.imshow(self.unload_tensor(self.content.data))

    plt.subplot(132)
    plt.imshow(self.unload_tensor(self.style.data))

    plt.subplot(133)
    plt.imshow(self.unload_tensor(self.result.data))

    plt.show()

  def single_combine(self):
    self.build_single_style_model()
    self.train_single_model()
    
    pass 

  def build_single_style_model(self):
    content = self.content.clone()
    style = self.style.clone()

    cnn = models.vgg19(pretrained=True).features

    self.content_losses = []
    self.style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model.forward(content).clone()
                content_loss = ContentLoss(target, self.content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                self.content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model.forward(style).clone()
                target_feature_gram = gram.forward(target_feature)
                style_loss = StyleLoss(target_feature_gram, self.style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                self.style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    print(model)
    self.model = model 

  def train_single_model(self):
    # input image
    input = self.content.clone()
    input.data = torch.randn(input.data.size()).type(torch.FloatTensor)

    input = nn.Parameter(input.data)
    optimizer = optim.LBFGS([input])

    run = [0]
    while run[0] <= 10:

        def closure():

            input.data.clamp_(0, 1)

            optimizer.zero_grad()
            self.model.forward(input)
            style_score = 0
            content_score = 0

            for sl in self.style_losses:
                style_score += sl.backward()
            for cl in self.content_losses:
                content_score += cl.backward()

            run[0]+=1
            if run[0] % 10 == 0:
                print("run " + str(run) + ":")
                print(style_score.data[0])
                print(content_score.data[0])

            return content_score+style_score

        optimizer.step(closure)

    input.data.clamp_(0, 1)
    self.result = input 


if __name__ == '__main__':
  content_layers = ['conv_4']
  style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
  
  content = '../input_image/content.jpg'
  style = '../input_image/style.jpg'
  
  model = StyleModel(content_layers, style_layers)
  model.transfer(content, style)


  
