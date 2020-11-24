# NCTU_CS_T0828_HW2-Street View House Numbers images object detection
## Introduction
The proposed challenge is a Street View House Numbers images object detection task using the SVHN dataset, which contains 33,402 trianing images, 13,068 test images.
Train dataset | Test dataset
------------ | ------------- |
33,402 images | 13,068 images
## Hardware
The following specs were used to create the original solution.
- Ubuntu 16.04 LTS
- Intel(R) Core(TM) i7-7500U CPU @ 2.90GHz
- 2x NVIDIA 2080Ti
## Data pre-process
### Get image & bounding box informations
Firstly, use the **construct_dataset.py** to read the **digitStruct.mat**, get the img_bbox_data.
```
$ python3 construct_dataset.py
```
And then, use the getimgdata.py to get the images width & height, merge w & h with img_bbox_data, get the all image data:
img_name | label | left | top | width | height | right | bottom | img_width | img_height
------------ | ------------- |
```
Trainin_data
  +- train
    |	+- label 1
    |	+- label 2
    | 	+- label 3 ....(total 196 species labels )
  +- val
    |	+- label 1
    |	+- label 2
    |   +- label 3 ....(total 196 species labels )
```
### Data augmentation
Since there are 196 kinds of cars to be trained, the training data may not be enough to cause overfit. Therefore, before input data into the model, we can generate more data for the machine to learn by means of data augmentation. 
```
transforms.Compose([
        transforms.Resize((450, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
## Training
### Model architecture
PyTorch provides several pre-trained models with different architectures. 

Among them, **ResNet152** is the architecture I adopted and I redefine the last layer to output 196 values, one for each class. As it gave the best validation accuracy upon training on our data, after running various architectures for about 10 epochs.
### Train models
To train models, run following commands.
```
$ python3 training.py
```
### Hyperparameters
Batch size=32, epochs=20
#### Loss Functions
Use the nn.**CrossEntropyLoss()**
#### Optimizer
Use the **SGD** optimizer with lr=0.01, momentum=0.9.

Use the lr_scheduler.**StepLR()** with step_size=10, gamma=0.1 to adjust learning rate. 
### Plot the training_result
Use the **plot.py** to plot the training_result

---training_data loss & acc, and validation_data loss & acc
```
$ python3 plot.py
```
## Testing
Using the trained model to pred the testing_data.
```
$ python3 test_data.py
```
And get the result, save it as csv file.
## Submission
Submit the test_result csv file, get the score.

The best resutl is 91.92%.
