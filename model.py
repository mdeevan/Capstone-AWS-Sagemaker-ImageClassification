


def net(num_classes, model_location=""):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    # model = models.resnet50(pretrained=True)
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    model = models.resnet50(pretrained=True)
    
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


    logger.info("Model creation completed")

    return model