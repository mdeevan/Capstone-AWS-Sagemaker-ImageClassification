#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms


import argparse

def get_pretained_model_ResNet50():
    weights = models.ResNet50_Weights()
    model = models.resnet50(weights=weights)

    return model


def test(model, test_loader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    pass

def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''


    epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    for e in epochs:

        running_loss = 0
        correct = 0
        
        for data, target in train_loader:
            data   = data.to_device(device)
            target = target.to_device(device)

            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss
        
            loss.backward()
            optimizer.step()

            pred = pred.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            total_loss = running_loss / len(train_loader.dataset)
            accuracy = correct / len(train_loader.dataset)

            print("epoch : {}, total loss : {}, accuracy :{}%".format(e, total_loss, accuracy))
            
    
    pass
    
def net(num_classes = 100):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''

    # model = torchvision.models.detection.ResNet50_Weights()
    model = get_pretained_model_ResNet50

    for params in model.parameters:
        params.requires_grad = False

    num_features = model.fc.in_features()
    # num_classes = 100

    model.fc = nn.Sequential(
        nn.Linear(num_features, num_classes)
    )

    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    pass

def main(args):

    
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = None
    optimizer = None
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    
    args=parser.parse_args()
    
    main(args)
