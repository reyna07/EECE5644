import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule
from captum.attr import Saliency

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # For M1 chip, torch.device("mps") is used instead of torch.device("cuda")



def explainer_improved(model, labels, DEVICE, img, true_label):
    
    model.to(DEVICE)
    model.eval()

    attribution = Saliency(model)
    

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transformed_img = transform(img)

    input = transform_normalize(transformed_img)
    input = input.unsqueeze(0)

    output = model(input)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_label = torch.max(probabilities, 0)[1].item()

    #pred_label_idx.squeeze_()
    #predicted_label = labels_human[str(pred_label_idx.item())][1]
    print('Predicted:', labels[predicted_label])

    # Compute the attribution scores using Saliency for true label
    attr_true = attribution.attribute(inputs=input, target=true_label)

    # Compute the attribution scores using Saliency for predicted label
    attr_pred = attribution.attribute(inputs=input, target=predicted_label)

    # Transform the image to original scale
    func = input * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    
    # Visualize the attribution scores for true label
    explainer_true, _ = torch.max(attr_true.data.abs(), dim=1) 
    explainer_true = explainer_true.cpu().detach().numpy()
    explainer_true = (explainer_true-explainer_true.min())/(explainer_true.max()-explainer_true.min())

    # Visualize the attribution scores for predicted label
    explainer_pred, _ = torch.max(attr_pred.data.abs(), dim=1)
    explainer_pred = explainer_pred.cpu().detach().numpy()
    explainer_pred = (explainer_pred-explainer_pred.min())/(explainer_pred.max()-explainer_pred.min())
    
    fig, ax = plt.subplots(1,3, figsize=(30,50))

    ax[0].imshow(transformed_img.permute(1, 2, 0).numpy())
    ax[1].imshow(explainer_true[0])
    ax[1].set_title(f"True: {labels[true_label][0]}", fontsize=48)
    ax[2].imshow(explainer_pred[0])
    ax[2].set_title(f"Predicted: {labels_human[predicted_label][0]}", fontsize=48)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    
    fig.subplots_adjust(wspace=0, hspace=0, top=1.0)
    FIG_PATH = "hw4\Saliency" + str(true_label) + ".png"
    plt.savefig(FIG_PATH, bbox_inches='tight')



if __name__=="__main__":
    # Create the model
    model_googlenet= models.googlenet(pretrained=True)

    # Load classes to human readable labels
    labels_human = {}
    with open(f'hw4\imagenet1000_clsidx_to_labels.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("'", "").strip(",")
            if "{" in line or "}" in line:
                continue
            else:
                idx = int(line.split(":")[0])
                lbl = line.split(":")[1].split(",")
                labels_human[idx] = [x.strip() for x in lbl]

    

    #true_label = 1 #this is the label number for the goldfish 
    #img = Image.open('hw4/a-67-pound-goldfish-named-the-carrot.jpg')
    #img = Image.open('hw4/goldfish.jpg')


    #true_label = 94
    #img = Image.open('hw4/hummingbird.jpg')

    #true_label = 100
    #img = Image.open('hw4/black-swans.jpg')

    #true_label = 207
    #img = Image.open('hw4/golden-ret.jpg')

    true_label = 985
    img = Image.open('hw4/papatya.jpg')

    explainer_improved(model_googlenet, labels_human, DEVICE, img, true_label)

