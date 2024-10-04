# Enhancing Document Image Binarization through Domain Adaptation with Transformer Cross-Attention

## Overview

This model tackles the tough problem of cleaning up degraded document images for better recognition and use. We introduce a groundbreaking solution: an encoder-decoder architecture powered by cross-attention vision transformers. This model promises to elevate the quality of both machine-printed and handwritten documents through end-to-end training. It cleverly learns both universal and domain-specific features to enhance the denoising process. Our encoder uses cross-attention to handle pixel patches and positional data directly, skipping convolutions, while the decoder adeptly rebuilds clear images from the processed patches. Extensive testing shows our model surpasses the best existing methods on various DIBCO benchmarks, heralding a new era in document image processing.

## Environment Install

Clone this repository and navigate to the root directory of the project.
```
conda create -n docbin python=3.8
conda activate docbin
git clone https://github.com/kunalpurkayastha/cadocbin.git
cd cadocbin
```
### Requirement
```
pip install -r requirement.txt
```
## Using CADOCBIN

### Data Arrangement
```
data
|_ train
|    |_ 1.jpg
|    |_ 2.jpg
|    ...
|_ train_gt
|    |_ 1.jpg
|    |_ 2.jpg
|    ...
|_ valid
|    |_ 1.jpg
|    |_ 2.jpg
|    ...
|_ valid_gt
|    |_ 1.jpg
|    |_ 2.jpg
|    ...
|_ test
|    |_ 1.jpg
|    |_ 2.jpg
|    ...
|_ test_gt
     |_ 1.jpg
     |_ 2.jpg
     ...
```

### Training
For training, specify the desired settings (batch_size, patch_size, model_size, split_size and training epochs) when running the file train.py. For example, for a base model with a patch_size of (16 X 16) and a batch_size of 32 we use the following command:

```
python train.py --data_path /YOUR_DATA_PATH/ --batch_size 32 --vit_model_size base --vit_patch_size 16 --epochs 151 --split_size 256 --validation_dataset 2016
```
You will get visualization results from the validation dataset on each epoch in a folder named vis+"YOUR_EXPERIMENT_SETTINGS" (it will be created). In the previous case it will be named visbase_256_16. Also, the best weights will be saved in the folder named "weights".

### Testing on a DIBCO dataset
To test the trained model on a specific DIBCO dataset (should be matched with the one specified in Section Process Data, if not, run process_dibco.py again). Download the model weights (In section Model Zoo), or use your own trained model weights. Then, run the following command. Here, I test on H-DIBCO 2018, using the Base model with 8X8 patch_size, and a batch_size of 16. The binarized images will be in the folder ./vis+"YOUR_CONFIGS_HERE"/epoch_testing/

It is recommended to test the model using GPU rather than CPU. To run using GPU install the corresponding torch version compiled with CUDA using the below command

```
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

```
Test the model on your dataset using the below command

```
python test.py --data_path /YOUR_DATA_PATH/ --model_weights_path  /THE_MODEL_WEIGHTS_PATH/ 
```

### Run Inference on custom images

Run the below command
```
python viz.py 
```

### Model Weights

The pre-trained weights for all the best CADOBIN model variants trained on DIBCO benchmarks are available at:


https://drive.google.com/drive/folders/1gaGC69AtLTvWSQo-VZciAMsV8C-EGDSj?usp=drive_link


### Dataset

The dataset used to train and test the model are available at: 

https://drive.google.com/file/d/1qsjVS1b8lkPKp9I39Mn2zX_u4fQlvnxz/view?usp=drive_link