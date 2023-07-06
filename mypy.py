import time

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import torchvision.models as models

import copy

from torch import squeeze


import telebot
from os import path, remove
from telebot import types

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
imsize = (512, 512) if torch.cuda.is_available() else (128, 128)  # use small size if no GPU


loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image_size = image.size
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float), image_size


unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()


class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.loss = None
        self.target = target.detach()

    def forward(self, inp):
        self.loss = F.mse_loss(inp, self.target)
        return inp


def gram_matrix(inp):
    a, b, c, d = inp.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = inp.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.loss = None
        self.target = gram_matrix(target_feature).detach()

    def forward(self, inp):
        G = gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target)
        return inp


cnn = models.vgg19(pretrained=True).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_losses(cnnf, normalization_mean, normalization_std,
                               style_imgf, content_imgf,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnnf.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_imgf).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_imgf).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_imgf):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_imgf])
    return optimizer


def run_style_transfer(cnnf, normalization_mean, normalization_std,
                       content_imgf, style_imgf, input_imgf, num_steps=700,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnnf,
                                                                     normalization_mean, normalization_std, style_imgf,
                                                                     content_imgf)

    input_imgf.requires_grad_(True)

    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_imgf)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_imgf.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_imgf)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_imgf.clamp_(0, 1)

    return input_imgf


def join_photo():
    style_img, _ = image_loader("./downloads/img2.jpg")
    content_img, content_size = image_loader("./downloads/img1.jpg")

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    input_img = content_img.clone()

    loader2 = transforms.Compose([
        transforms.Resize(content_size)])

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    output = loader2(output)
    output = output.squeeze(0)
    output = unloader(output)
    return(output)

token = "5911525993:AAH7H9x9pF-qANqDDLXdbvXLRQQBdjkkZWI"
bot = telebot.TeleBot(token, parse_mode=None)


@bot.message_handler(commands=['help'])
def send_tusk(message):
    bot.send_message(message.chat.id, "Я - бот, который может передать стиль одной фотографии - другой. Чтобы начать работу, пришлите фотографию, на которую хотите перенести стиль")


@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src1 = "downloads/" + "img1." + file_info.file_path[-3:]
    src2 = "downloads/" + "img2." + file_info.file_path[-3:]
    if not path.exists(src1):
        with open(src1, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "Фотография получена. Теперь отправте фотографию, стиль которой хотите передать первой")
    else:
        with open(src2, 'wb') as new_file:
            new_file.write(downloaded_file)
        bot.reply_to(message, "Фотография получена. Начинаю творить, ждите...")
        res = join_photo()
        bot.send_photo(message.chat.id, res)
        remove(src1)
        remove(src2)


@bot.message_handler(func=lambda message: True)
def send_welkom(message):
    bot.send_message(message.chat.id, "Приветствую! Напишите команду /help, чтобы узнать, что я могу")


bot.infinity_polling()
