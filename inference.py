#Reference: https://sagemaker-examples.readthedocs.io/en/latest/frameworks/pytorch/get_started_mnist_deploy.html
#Reference: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import os
import json
import io
from PIL import Image
import logging
import sys
import os
from flask import Flask, request, jsonify


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



app = Flask(__name__)


def Net(num_classes):
    print("inference: model creation")
    model = models.resnet50(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False   

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
                    nn.Linear(num_features, num_classes),
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
    # logger.info(f'Inference: input function with content type :{request_content_type} \n and request {request_body}')
    # assert request_content_type=='image/jpeg'
    # return Image.open(io.BytesIO(request_body))


    logger.info('Deserializing the input data.')
    
    # process an image uploaded to the endpoint
    
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    
    if content_type == JPEG_CONTENT_TYPE: 
        return Image.open(io.BytesIO(request_body))
    
    logger.debug('Loaded JPEG content')
    
    # process a URL submitted to the endpoint
    
    if content_type == JSON_CONTENT_TYPE:
        #img_request = requests.get(url)
        logger.debug(f'Request body is: {request_body}')

        request = json.loads(request_body)

        logger.debug(f'Loaded JSON object: {request}')

        url = request['url']
        img_content = requests.get(url).content

        return Image.open(io.BytesIO(img_content))
    
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))


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
    res = np.argmax(np.asarray(res))
    
    return json.dumps(res)


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'healthy'}), 200

@app.route('/invocations', methods=['POST'])
def invoke():
    # Get the content type of the request
    request_content_type = request.content_type
    response_content_type = 'application/json'  # Set the response content type

    # Call input_fn to prepare the input data
    input_data = input_fn(request.get_json(), request_content_type)

    # Call predict_fn to get the predictions
    predictions = predict_fn(input_data, model)

    # Call output_fn to prepare the response
    response = output_fn(predictions, response_content_type)

    return response



    data = request.get_json()
    input_tensor = torch.tensor(data['input'])  # Adjust based on your input format
    # with torch.no_grad():
    #     output = model(input_tensor)

    response = output.numpy().tolist()  # Convert output to a list for JSON serialization
    return jsonify({'predictions': response}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)