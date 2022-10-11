

import warnings
warnings.filterwarnings("ignore")
import time
import pixellib
from pixellib.custom_train import instance_custom_training

### Train a custom model using your dataset
##This method adapts the coco model

train_maskrcnn = instance_custom_training()
train_maskrcnn.modelConfig(network_backbone = "resnet50", num_classes= 1, batch_size = 6)
train_maskrcnn.load_pretrained_model("Models/mask_rcnn_coco.h5")
train_maskrcnn.load_dataset("Sub_Mountain_USEME")
print('starting model training for mountain object')
start = time.process_time()
train_maskrcnn.train_model(num_epochs = 10, augmentation=True,  path_trained_models = "mask_rcnn_models_resnet50")
print(time.process_time() - start)
      
print("DONE!")