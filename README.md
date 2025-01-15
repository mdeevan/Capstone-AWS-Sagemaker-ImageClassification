# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

Candidate pre-trained algorithms for the capstone
ResNet, Inception, VGG and MobileNet 

resnet50 was fine tuned by adding additional fully-connected layer and softmax. Weights were frozen in all layers except the newly added layers for fine-tuning.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 


## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.  

```python
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip --no-check-certificate
```


### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
s3 location : "s3://sagemaker-us-east-1-559647231942/sagemaker/Image_classification"

```python

s3_data_location = sagemaker_session.upload_data(path="dogImages", bucket=bucket, key_prefix=prefix)

```

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Remember that your README should:
![training jobs](./screenshots/Training%20Jobs%20-%20SageMaker%20Studio.png)  
![training jobs](./screenshots/Training%20Jobs.png)  

![hyperparamter tuning jobs](./screenshots/Hyperparamter%20tuning%20jobs.png)


- Logs metrics during the training process
- Tune at least two hyperparameters

Three parameter were tuned, learning-rate, batch-size and epochs
```python
hyperparameter_ranges = {
    "lr"         : ContinuousParameter(0.001, 0.1),
    "batch-size" : CategoricalParameter([16, 32, 64 ]),
    "epochs"     : IntegerParameter(10,  20)
}

objective_metric_name = "Accuracy"
objective_type = "Maximize"
metric_definitions = [{"Name": "Accuracy", "Regex": "Test set: Accuracy: ([0-9\\.]+)"}]
```

- Retrieve the best best hyperparameters from all your training jobs

![Best parameters](./screenshots/best%20hyperparameters.png)  


## Debugging and Profiling
Give an overview of how you performed model debugging and profiling in Sagemaker

### Results
What are the results/insights did you get by profiling/debugging your model?
![Debug output](./screenshots/train%20vs%20eval.png)


Remember to provide the profiler html/pdf file in your submission.

[link to the profile output](./ProfilerReport/profiler-output/profiler-report.pdf)

## Model Deployment
Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![Deployed endpoint](./screenshots/inference%20Endpoints%20-%20SageMaker%20Studio.png)
![Deployed endpoint summary](./screenshots/inference%20endpoint%20-%20summary.png)


## Standout Suggestions
This is where you can provide information about any standout suggestions that you have attempted.

#### Batch Transformation

![Batch Transformation](./screenshots/Batch%20transform%20jobs%20Amazon%20SageMaker%20AI%20us-east-1.png)
![S3 Input folder](./screenshots/S3%20bucket%20-%20batch%20transformation%20-input.png)
![S3 output folder](./screenshots/S3%20bucket%20-%20batch%20transformation%20-output.png)

#### Docker Image

Image was successfully created. However, the lack of quota to copy image to ECR, prevents from testing the image  

