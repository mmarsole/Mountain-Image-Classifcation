# Mountain-Image-Classifcation

## Simple summary: 

The Goal of this project was to train a CNN model to recognitize Mountains from imgery at the pixel level (segmantic segmentation).

## File summary

This repository contains data, files, and models for this project Mountain-Image-Classification (an attempt at training a model to segment mountains from an image). 

The folder for this project "MountainIDProjectSubmission", contains three folders, "MSI_Files", "JupNotebooks_PCenv", "Data". Within the "MSI_Files" folder are the files used to run jobs via the [Minnesota Supercomputing Insitute (MSI)](https://www.msi.umn.edu/) portal ("SchedulingJob_50.sh" and "Mountain_model_training_MSI_50.py"). Operating in MSI, involved making a python environment with additional python packages beyond those included in conda (`imgaug`, `tensorflow`, and `pixellib`). In addition to downloading packages I also had to install several dependencies for older versions (`tensorflow==1.5.0`, `skimage == 0.16.2`, and `keras==2.1.5`) to avoid certain errors. "slurm-9072506.out" is one example (1 epoch) of results from running the files in MSI. 

In "JupNotebooks_PCenv" you will find the Jupyter Notebooks used to run the code on my personal PC environment (this too required downloading the previously  mentioned packages when operating in a MSI environment). It should be noted that MSI was utilized solely to train models, while the Jupyter Notebooks executed on my personal PC explored evaluating the outcomes (visualizations and MAP calculations) in addition to training. 

Lastly, the "Data" folder contains the image dataset used to train and test the models on classifying mountains (309 train images and 105 test images all with corresponding json files), while the "Models" folder contains the model adapted (mask_rcnn_coco.h5) and all resulting models from my training ("Resulting Models", each model was saved after each epoch involved in training it). 

### Note before running any file. 

If you desire to run any files, you will need to adapt the Path to current locations (such as the path to access the "Sub_Mountain_USEME" dataset or the "mask_rcnn_coco.h5" model used to train the data. 

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
### Data/
Sub_Mountain_small:
  * A sample of the original data used to train the model.
  * Contains 18 images (in test and train folders)
#### Models/OriginalModel
  Mask_rcnn_coco.h5:
   * A pretrained model, built upon to train a model to identify Mountains
#### Models/ResultingModels
  Mask_rcnn_model.001-2.025847.h5: (not provided)
   * Trained model after 1 epoch
  
  Mask_rcnn_model.002-1.948599.h5:(not provided)
   * Trained model after 2 epochs
  
  Mask_rcnn_model.004-1.862559.h5:(not provided)
   * Trained model after 4 epochs
  
  Mask_rcnn_model.005-1.854457.h5:(not provided)
   * Trained model after 5 epochs
  
  Mask_rcnn_model.006-1.754546.h5: (supplied)
   * Trained model after 6 epochs
*Please note, the largest files within this ZIpfile would be the models (each vary around 175,000KB)

### Citations:
Mountain Images were provided by [MIT Computer Science and Artificial Intelligence Laboratory](http://places.csail.mit.edu/) "Places" dataset. 

B. Zhou, A. Lapedriza, J. Xiao, A. Torralba, and A. Oliva. “Learning Deep Features for Scene Recognition using Places Database.” Advances in Neural Information Processing Systems 27 (NIPS), 2014. PDF Supplementary Materials

Most code was based upon the tutorial ["Custom Instance Segmentation Training With 7 Lines Of Code."](https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b)

Olafenwa (she/her), A. (2020, November 29). Custom Instance Segmentation Training With 7 Lines Of Code. Medium. https://towardsdatascience.com/custom-instance-segmentation-training-with-7-lines-of-code-ff340851e99b
