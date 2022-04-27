# SensingGAN: A Lightweight GAN for Real-time Rain Removal with Self-attention
### Author: [Chun-Po Chen](https://www.linkedin.com/in/chenabei0421/)
### Advisor: [Pao-Ann Hsiung](https://www.cs.ccu.edu.tw/~pahsiung/introduction/biography.html)
### Link: [Online Speech: Pao-Ann Hsiung](https://www.youtube.com/watch?v=onE_wNJvLes)

## Abstract
![The example of the detection result of rain removal for object detection.](/Figures/Example_Derained_ObjectDetection.png "The example of the detection result of rain removal for object detection.")
Rain may degrade the quality of source images for the application of Computer Vision. For example, the object detection system of autonomous vehicles may be inaccurate due to rain. We proposes SensingGAN: a lightweight Generative Adversarial Network (GAN)-based Single Image De-raining method. There are 2 main challenges to meet the needs of computer vision applications for rain removal:
1. The real-world rain is diverse, so it is difficult to extract rain using a simple method, and it is not easy to restore the edges and details of objects covered by rain after de-raining.
1. In the past, many methods focused on the de-raining performance, but the use of complex architectures could not meet the needs of the real-time environment in terms of efficiency.
Therefore, we discusses how to achieve a better balance between de-raining performance and efficiency, which can provide high-quality de-rained images for computer vision in the Rain in Driving (RID) dataset.

![The architecture of SensingGAN.](/Figures/SensingGAN.png "The architecture of SensingGAN.")
SensingGAN can effectively sense objects and rain like humans, and restore the details of objects to satisfy the high safety and efficiency requirements of autonomous vehicles. SA-Feature Loss can not only maintain the efficiency but also can more clearly distinguish objects to restore the details and shapes of objects. The loss function and discriminator improve the de-raining performance in the training stage without requiring extra execution time. SensingGAN increases object detection (YOLO V4-Tiny) accuracy by 3% in RID. In comparison with classical de-raindrop GAN, FPS is improved by 13 times (10 ms).

## Experimental Results
### Ablation Study (Trained with Raindrop)
![The ablation test results of SensingGAN in raindrop situation.](/Figures/AblationRaindrop.png "The ablation test results of SensingGAN in raindrop situation.")
### Ablation Study (Trained with Rain100H)
![The ablation test results for SensingGAN in rain streaks situation.](/Figures/AblationRaindrop.png "The ablation test results for SensingGAN in rain streaks situation.")
### Raindrop Removal
![Compare PSNR, SSIM, FPS in real raindrop with other methods.](/Figures/RaindropRemoval.png "Compare PSNR, SSIM, FPS in real raindrop with other methods.")
### Raindrop Removal
![The examples in heavy rain with other methods.](/Figures/RainStreaksRemoval.png "The examples in heavy rain with other methods.")
### Object Detection (YOLO V4-Tiny, RID)
* YOLO V4-Tiny (AlexeyAB)
* Classes: Car, Person, Bus, Motorbike, Bicycle
![The results of object detection by de-rained results of comparison methods.](/Figures/ObjectDetectionAfterRainRemoval.png "The results of object detection by de-rained results of comparison methods.")

## Requirements
* Jupyter
* Python 3.5
* Pytorch 1.0

## Datasets

## Folders
* rainy_image_dataset: Dataset
* models: Trained models
* data: PSNR、SSIM results
* results: De-rained images results
* samples: De-rained images in training
* pytorch_ssim: Calculate SSIM

## Train
**train_own.ipynb**: Click **Run** on Juypter to start training after adjusting architecute of model and training parameters.

* Architecute of model:
    * **network.py: Architecutre of SensingGAN**
    * loss_function: Architecutre of SA-Feature Loss
    * dataset.py: Dataset execution
    * trainer.py: Scale of loss function
    * utils.py: Load dataset in get_files
    * spectral.py: Spectral Normalization

* Training parameters:
    * os.environ["CUDA_VISIBLE_DEVICES"]: Run GPU
    * **save_path: Save path of model**
    * **baseroot: Path of Training Data**
    * **train_batch_size: Batch Size**
    * **epochs: Training Epochs**
    * sample_path: Save images in training
    * save_by_epoch: Save the model every few Epochs
    * lr: Learning Rate
    * b1: Beta1 of Adam
    * b2: Beta2 of Adam

## Test
**test.ipynb**: Click **Run** on Juypter to start testing after adjusting architecute of model and training parameters.

* Architecute of model:
    * **network.py: The architecutre of load_gname model**
    * utils.py: Load dataset in get_files

* Testing parameters
    * os.environ["CUDA_VISIBLE_DEVICES"]: Run GPU
    * **save_name: Save path of de-rained results**
    * **load_gname: Load path of model**
    * **baseroot: Load path of testing Data**
    * resize: Is adjuct image-size
    * scale_size: Max image-size