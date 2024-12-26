#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse 
import logging
import sys
import os
# import smdebug

# import smdebug.Pytorch as smd

# from smdebug import modes
# from smdebug.profiler.utils import str2bool
# from smdebug.pytorch import get_hook



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def test(model, test_dataloader, criterion, device=torch.device("cpu")):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    logger.info('HPO: model test starting')

    model.eval()
    with torch.no_grad():
        test_running_loss = 0
        accuracy_running = 0

        for data, target in test_dataloader:
            data   = data.to(device)
            target = target.to(device)

            pred = model(data)
            loss = criterion(pred, target)
            test_running_loss += loss.item()

            prob = torch.exp(pred)
            top_label, top_class = prob.topk(1, dim=1)

            correct = top_class == target.view(*top_class.shape)
            accuracy_running += torch.mean(correct.type(torch.FloatTensor)).item()

        test_loss = test_running_loss / len(test_dataloader)
        accuracy  = accuracy_running  / len(test_dataloader)

        logger.info("HPO: Test loss : {:.3f},\
            Test set: Accuracy: {:.3f} ".format(test_loss,
                                        accuracy
                                        ))


        logger.info("HPO: model testing completed")

def train(model, train_dataloader, valid_dataloader, criterion, optimizer, epochs=2, device=torch.device("cpu")):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''

    train_loss = []
    valid_loss = []
    accuracy   = []
    prev_accuracy = 0

    logger.info('HPO: Training started on device {}'.format(device))

  
    for epoch in range(epochs):

        train_running_loss = 0
        valid_running_loss = 0
        accuracy_running   = 0
        
        model.train()
        for data, target in train_dataloader:
            data   = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            pred = model(data)
            loss = criterion(pred, target)
            train_running_loss += loss.item()
        
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_running_loss = 0

            for data, target in valid_dataloader:
                data   = data.to(device)
                target = target.to(device)

                pred = model(data)
                loss = criterion(pred, target)
                valid_running_loss += loss.item()

                prob = torch.exp(pred)
                top_label, top_class = prob.topk(1, dim=1)

                correct = top_class == target.view(*top_class.shape)
                accuracy_running += torch.mean(correct.type(torch.FloatTensor)).item()

        # training and evaluation loop completed

        train_running_loss = train_running_loss / len(train_dataloader)
        valid_running_loss = valid_running_loss / len(valid_dataloader)
        accuracy_running   = accuracy_running   / len(valid_dataloader)


        train_loss.append(train_running_loss)
        valid_loss.append(valid_running_loss)
        accuracy.append(accuracy_running)

        logger.info("HPO: epoch {} of {},\
            training loss : {:.3f},\
            validation loss : {:.3f},\
            accuracy : {:.3f} ".format(epoch+1, epochs,
                                            train_running_loss,
                                            valid_running_loss,
                                            accuracy_running
                                            ))


    return model    

def net(num_classes):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # model = models.resnet50(pretrained=True)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model = models.resnet18(pretrained=True)
    
    for params in model.parameters():
        params.requires_grad = False

    num_features = model.fc.in_features

    # model.fc = nn.Sequential(
    #                 nn.Linear(num_features, 1024),
    #                 nn.ReLU(),
    #                 nn.Linear(1024        , 512),
    #                 nn.ReLU(),
    #                 nn.Linear(512         , num_classes),
    #                 nn.Softmax(dim=1)
    #                 )
    model.fc = nn.Sequential(
                    nn.Linear(num_features, 512),
                    nn.ReLU(),
                    nn.Linear(512         , num_classes),
                    nn.Softmax(dim=1)
                    )


    logger.info("HPO: Model creation completed")
    return model

def create_data_loaders(data_train, data_valid, data_test, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    logger.info("HPO: creating data loaders")
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    training_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])


    trainset = torchvision.datasets.ImageFolder(data_train, transform=training_transform)
    validset = torchvision.datasets.ImageFolder(data_valid, transform=testing_transform)
    testset  = torchvision.datasets.ImageFolder(data_test , transform=testing_transform)

    logger.info("HPO: Batch Size {}".format( batch_size))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(testset , batch_size=batch_size, shuffle=True)

    logger.info('HPO: Data loaders created')
    return train_loader, valid_loader, test_loader


def main(args):


    logger.info("arguments {}".format(args))
    logger.info(f'HPO: Hyperparameters are LR: {args.lr}, Batch Size: {args.batch_size}')

    '''
    TODO: Initialize a model by calling the net function
    '''

    device = torch.device("cuda" if (torch.cuda.is_available() & args.gpu) else "cpu")
    logger.info(f"HPO: Running on Device {device}")

    logger.info(f"HPO: Retrieving model")
    model=net(args.num_classes)
    model.to(device)


    logger.info('HPO: create the data loaders')
    train_loader, valid_loader, test_loader=create_data_loaders(args.data_train,  args.data_valid,args.data_test, args.batch_size )


    '''
    TODO: Create your loss and optimizer
    '''

    loss_criterion = nn.NLLLoss() # using negative log likelihood loss 

    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr) #using adam optimizer

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    logger.info('HPO: train the model')
    model=    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, device)


    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("HPO: Test the Model")
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    logger.info("HPO: Saving Model")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth")) # save the trained model to S3

    logger.info("HPO: Script completed, exiting")



if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
#   https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md

    parser.add_argument('--data-train', type=str,      default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--data-test',  type=str,      default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--data-valid', type=str,      default=os.environ['SM_CHANNEL_VAL'])
    parser.add_argument('--model-dir',  type=str,      default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument("--num_classes",type=int,      default=os.environ['NUM_CLASSES']) #default=10,   metavar="N", help="Number of classes for classification (default = 10)")
    # parser.add_argument("--gpu",        type=str2bool, default=True, metavar="N", help="Train on GPU, (default = True)")

    # parser.add_argument('--output-dir', type=str,      default=os.environ['SM_OUTPUT_DATA_DIR'])
    # parser.add_argument("--data-path",  type=int,      default=os.environ['SM_CHANNEL_TRAINING'],   metavar="N", help="S3 location of train/test/valid data")

    parser.add_argument("--batch-size", type=int,      default=64,   metavar="N", help="input batch size for training (default: 64)",)
    parser.add_argument("--epochs",     type=int,      default=1,    metavar="N", help="number of epochs to train (default: 1)") 
    parser.add_argument("--lr",         type=float,    default=0.05, metavar="N", help="Learning rate (default = 0.05)")
    parser.add_argument("--num-classes",type=int,      default=133,  metavar="N", help="Number of classes")
    parser.add_argument("--gpu",        type=bool,     default=True, metavar="N", help="gpu training, default = True")

    args=parser.parse_args()
    
    main(args)
