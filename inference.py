#Reference: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# from torchvision.models import ResNet50_Weights
import os
import json
import io
from PIL import Image
import logging


def Net(num_classes):
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512         , num_classes),
                    nn.Softmax(dim=1)
                    )
                    
    return model

def model_fn(model_dir):
    model = Net(133)

    logger.info('Inference: Model created')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    logger.info('Inference: Model loaded')

    return model


def input_fn(request_body, request_content_type):
    assert request_content_type=='image/jpeg'
    return Image.open(io.BytesIO(request_body))


def predict_fn(input_data, model):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        input_data = testing_transform(input_data).unsqueeze(0).to(device)
        predictions = model(input_data)

    return predictions


# def output_fn(predictions, content_type):
#     assert content_type == 'application/json'

#     res = predictions.cpu().numpy().tolist()
#     return json.dumps(res)