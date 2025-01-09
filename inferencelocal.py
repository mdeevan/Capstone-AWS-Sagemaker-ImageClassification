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
import sys
import os


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def Net(num_classes):
    print("inference: model creation")
    model = models.resnet18(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512         , num_classes),
                    nn.Softmax(dim=1)
                    )
                    
    print("inference: model created")
    return model

def model_fn(model_dir):
    model = Net(133)

    logger.info('Inference: inside model_fn')
    print(f'inference: inside model_fn {model_dir}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f, map_location= device) )


    model.to(device).eval()

    logger.info('Inference: Model fn completed')
    print("model fn completed")

    return model


def input_fn(request_body, request_content_type):
    logger.info(f'Inference: input function with content type :{request_content_type} \n and request {request_body}')
    assert request_content_type=='image/jpeg'
    return Image.open(io.BytesIO(request_body))


def predict_fn(input_data, model):

    logger.info(f'Inference: predict function with input data :{input_data} ')
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

    logger.info(f'Inference: returning prediction :{predictions} ')
    return predictions


def output_fn(predictions, content_type):
    
    assert content_type == 'application/json'

    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)