# Mountain-Image-Classifcation

## Simple summary: 

The Goal of this project was to train a CNN model to recognize Mountains from imagery at the pixel level (semantic segmentation).

## File summary

This repository contains data, files, and models for this project Mountain-Image-Classification (an attempt at training a model to segment mountains from an image). 

This project contains three folders, "MSI_Files", "JupNotebooks_PCenv", "Sub_Mountain_small". Within the "MSI_Files" folder are the files used to run jobs via the [Minnesota Supercomputing Insitute (MSI)](https://www.msi.umn.edu/) portal ("SchedulingJob_50.sh" and "Mountain_model_training_MSI_50.py"). Operating in MSI, involved making a python environment with additional python packages beyond those included in conda (`imgaug`, `tensorflow`, and `pixellib`). In addition to downloading packages I also had to install several dependencies for older versions (`tensorflow==1.5.0`, `skimage == 0.16.2`, and `keras==2.1.5`) to avoid certain errors. "slurm-9072506.out" is one example (1 epoch) of results from running the files in MSI. 

In "JupNotebooks_PCenv" you will find the Jupyter Notebooks used to run the code on my personal PC environment (this too required downloading the previously  mentioned packages when operating in a MSI environment). It should be noted that MSI was utilized solely to train models, while the Jupyter Notebooks executed on my personal PC explored evaluating the outcomes (visualizations and MAP calculations) in addition to training. 

Lastly, the "Sub_Mountain_small" folder contains the image dataset used to train and test the models on classifying mountains (309 train images and 105 test images all with corresponding json files). Note, the resulting models (saved as h5 files) could not be shared on GitHub since their file size is too large. You can find the initial model (mask_rcnn_coco.h5) through this [article](https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b) and all resulting models from my training ("Resulting Models", each model was saved after each epoch involved in training it). 

### Note before running any file. 

If you desire to run any files, you will need to adapt the Path to current locations (such as the path to access the "Sub_Mountain_small" dataset or the "mask_rcnn_coco.h5" model used to train the data. 

If you do plan to rerun any notebooks, please note that the cell for training the model will take hours to execute (this has been an ongoing problem, trying to reduce the training time to more reasonable time constraints). 

### Recommendations (Future Work)

The code used to train the the model to identify mountains is based on the [tutorial](https://pixellib.readthedocs.io/en/latest/Image_instance.html) provided by the PixelLib package, which conveniently simplifies the process of deep learning for anyone not familiar with using keras or mask RCNNs. Based on the demo, results are better with more epochs (the number of complete iterations the model has had through the entire dataset), and is demonstrated using 300. One of the shortcomings of my project results currently has been trying to compute enough epochs to attain a better trained model. As it is, I have only successfully trained up to 6 epochs ( a fraction of the demonstrated 300) due to time restraints (it takes an average of 4 hours to complete one epoch, at this rate it would take 50 days to train a model up to 300 epochs). 

For the future, I'd recommend trying to run the code on a GPU environment (through MSI) to see if the training will be any faster. Additionally, I came across further documentation within PixelLib that demonstrates training with the Pytorch package (instead of the current tensorflow/keras package demoed in these files) results in better segmentation, and would thus try implementing code that trains with Pytroch instead of tensorflow.  

## File Index:
### MSI_Files/
Mountain_model_training_MSI_50.py:
  * file submitted to MSI (contains code only for training coco model)

SchedulingJob_50.sh:
  * Command script that submits the Mountain_model_training_MSI_50.py file to MSI

slurm-9072506.out:
  * Resulting output after running Mountain_model_training_MSI_50.py on MSI (1 epoch)
### JupNotebooks_PCenv
Mountain_training_test2_colabToPc-OG.ipynb:
  * This is the notebook first ran, its results I presented in class
  * (Contains code that trains, elvaluates, and visualizes output (after each trained model)

Mountain_training_test2_colabToPc.ipynb:
  * Contains code that trains, elvaluates, and visualizes output (after each trained model)

### Sub_Mountain_small:
  * A sample of the original data used to train the model.
  * Contains 18 images (in test and train folders)
#### Models/OriginalModel
  Mask_rcnn_coco.h5:
   * A pretrained model, built upon to train a model to identify Mountains
   * Obtain this model from [here](https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b). You will find the link within the  "FINAL STEP"
   * Please note, the models were to large to add directly to Git Hub (each vary around 175,000KB)
## Resources

Throughout this Project I collected links to helpful documents and files. Some used more readily than others: Here is a list of links I found relevant to learning and using PixeLIb to train the mask_rcnn_coco.h5 model:

Packages: 
* [PixelLib on GitHub](https://github.com/ayoolaolafenwa/PixelLib)
* [How to install and do Annotation of Images using Labelme (Easy, Simple & Flexible)](https://www.youtube.com/watch?v=XKJc2YT5-es)
* [LabelMe Annotation Tool](https://github.com/CSAILVision/LabelMeAnnotationTool)
    * Tool used to label mountain images with corresponding json files
    
Model: 
* [MaskR-CNN](https://github.com/matterport/Mask_RCNN)
    * helpful information about the model and deep learning used to train instance segmentation within PixelLib
* [Deep Residual Networks (ResNet, ResNet50) – Guide in 2021](https://viso.ai/deep-learning/resnet-residual-neural-network/)
    * This was a helpful document that explains the structure and architecture of the rcnn model (within PixelLIb can be either 'resnet50' or 'restnet101')
* [COCO model](https://arxiv.org/pdf/1405.0312.pdf) 
    * the coco model is a pretrained model that can identify 80 things, and is continually the backbone to furture catergorical traing, this document provides and interduction to the COCO model (used within PixelLib)
### Citations:
Mountain Images were provided by [MIT Computer Science and Artificial Intelligence Laboratory](http://places.csail.mit.edu/) "Places" dataset. 

B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. “Learning Deep Features for Scene Recognition using Places Database.” Advances in Neural Information Processing Systems 27 (NIPS), 2014. PDF Supplementary Materials

Most code was based upon the tutorial ["Custom Instance Segmentation Training With 7 Lines Of Code."](https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b)

Olafenwa (she/her), A. (2020, November 29). Custom Instance Segmentation Training With 7 Lines Of Code. Medium. https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b
