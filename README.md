# SensingGAN
### Author: 
### Link:

## Abstract


## Requirements
- Jupyter
- Python
- Pytorch

## Dataset

## Folders
- rainy_image_dataset: Dataset
- models: Trained models
- data: PSNR、SSIM results
- results: De-rained images results
- samples: De-rained images in training
- pytorch_ssim: Calculate SSIM

## Train
-train_own.ipynb: Click **Run** on Juypter to start training after adjusting architecute of model and training parameters.

    - Architecute of model:
        - **network.py: Architecutre of SensingGAN**
        - loss_function: Architecutre of SA-Feature Loss
        - dataset.py: Dataset execution
        - spectral.py: Spectral Normalization
        - trainer.py: Scale of loss function
        - utils.py: Load dataset in get_files

    - Training parameters:
        - os.environ["CUDA_VISIBLE_DEVICES"]: Run GPU
        - **save_path: Save path of model**
        - **baseroot: Path of Training Data**
        - **train_batch_size: Batch Size**
        - **epochs: Training Epochs**
        - sample_path: Save images in training
        - save_by_epoch: Save the model every few Epochs
        - lr: Learning Rate
        - b1: Beta1 of Adam
        - b2: Beta2 of Adam

## Test
-test.ipynb: Click **Run** on Juypter to start testing after adjusting architecute of model and training parameters.

    - Architecute of model:
        - **network.py: The architecutre of load_gname model**
        - utils.py: Load dataset in get_files
    
    - Testing parameters
        - os.environ["CUDA_VISIBLE_DEVICES"]: Run GPU
        - **save_name: Save path of de-rained results**
        - **load_gname: Load path of model**
        - **baseroot: Load path of testing Data**
        - resize: Is adjuct image-size
        - scale_size: Max image-size

## Contact