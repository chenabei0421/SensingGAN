# SensingGAN: A Lightweight GAN for Real-time Rain Removal with Self-attention
### Author: [Chun-Po Chen](https://www.linkedin.com/in/chenabei0421/), Advisor: [Pao-Ann Hsiung](https://www.cs.ccu.edu.tw/~pahsiung/introduction/biography.html)
### Link: [Online Speech: Pao-Ann Hsiung](https://www.youtube.com/watch?v=onE_wNJvLes)

## Abstract
![The example of the detection result of rain removal for object detection.](/Figures/Example_Derained_ObjectDetection.png "The example of the detection result of rain removal for object detection.")
Rain may degrade the quality of source images for the application of Computer Vision. For example, the object detection system of autonomous vehicles may be inaccurate due to rain. We proposes SensingGAN: a lightweight Generative Adversarial Network (GAN)-based Single Image De-raining method with **Self-attention**. There are 2 main challenges to meet the needs of computer vision applications for rain removal:
1. The real-world rain is diverse, so it is difficult to extract rain using a simple method, and it is not easy to restore the edges and details of objects covered by rain after de-raining.
1. In the past, many methods focused on the de-raining performance, but the use of complex architectures could not meet the needs of the real-time environment in terms of efficiency.

Therefore, we discusses how to achieve **a better balance between de-raining performance and efficiency**, which can provide high-quality de-rained images for computer vision in the Rain in Driving (RID) dataset.

### Architecture
![The architecture of SensingGAN.](/Figures/SensingGAN.jpg "The architecture of SensingGAN.")
**SensingGAN can effectively sense objects and rain like humans**, and restore the details of objects to satisfy the high safety and efficiency requirements of autonomous vehicles. **SA-Feature** Loss can not only maintain the efficiency but also can more clearly distinguish objects to restore the details and shapes of objects. The loss function and discriminator improve the de-raining performance in the training stage without requiring extra execution time. SensingGAN increases object detection (YOLO V4-Tiny) accuracy by **3%** in RID. In comparison with classical de-raindrop GAN, FPS is improved by **13** times (10 ms).

### SA-Feature Loss 
The loss of relations of feature values obtained by a pair of compared images applied by a VGG16, allowing the model to consider relations of high level features during training.
* Low-level details: relu2_1, relu2_2
* High-level features: SA(relu5_3)

## Experimental Results
### Datasets
* Rain100H: Heavy Rain Streaks | Synthetic
* Rain1400: Low/Medium Rain Streaks | Synthetic
* Raindrop: Raindrop | Real
* Rain in Driving (RID): Rain Streaks + Raindrop | Real

### Metrics
* Frame per Second (FPS): Speed
* Peak Signal-to-Noise Ratio (PSNR): The degree of noise
* Structural Similarity Index Measure (SSIM): Similarity of Luminance, Contrast, and Structure
* Mean Average Precision (mAP): Accuracy of object detection

### Ablation Study (Trained with Raindrop)
![The ablation test results of SensingGAN in raindrop situation.](/Figures/AblationRaindrop.png "The ablation test results of SensingGAN in raindrop situation.")
### Ablation Study (Trained with Rain100H)
![The ablation test results for SensingGAN in rain streaks situation.](/Figures/AblationRaindrop.png "The ablation test results for SensingGAN in rain streaks situation.")
### Raindrop Removal (Raindrop dataset)
![Compare PSNR, SSIM, FPS in real raindrop with other methods.](/Figures/RaindropRemoval.png "Compare PSNR, SSIM, FPS in real raindrop with other methods.")
### Raindrop Removal (Rain100H dataset)
![The examples in heavy rain with other methods.](/Figures/RainStreaksRemoval.png "The examples in heavy rain with other methods.")
### Object Detection (YOLO V4-Tiny, RID)
* [YOLO V4-Tiny (AlexeyAB)](https://github.com/AlexeyAB/darknet)
* Classes: Car, Person, Bus, Motorbike, Bicycle
![The results of object detection by de-rained results of comparison methods.](/Figures/ObjectDetectionAfterRainRemoval.png "The results of object detection by de-rained results of comparison methods.")

## Requirements
* Jupyter
* Python 3.5
* Pytorch 1.0

## Folders
* rainy_image_dataset: Dataset
* models: Trained models
* data: PSNR, SSIM results
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

## Main Reference
* Bdd100k: F. Yu, W. Xian, Y. Chen, F. Liu, M. Liao, V. Madhavan, and T. Darrell, “Bdd100k: A diverse driving video database with scalable annotation tooling,” arXiv preprint arXiv:1805.04687, May 2018
* M. Hnewa and H. Radha, “Object detection under rainy conditions for autonomous vehicles: A review of state-of-the-art and emerging techniques,” IEEE Signal Processing Magazine, vol. 38, no. 1, pp. 53–67, January 2021.
* S. Sundararajan, I. Zohdy, and B. Hamilton, “Vehicle automation and weather: Challenges and opportunities,” https://rosap.ntl.bts.gov/view/dot/32494/ dot_32494_DS1.pdf, December 2016.
* AttentiveGAN, Raindrop dataset: R. Qian, T. Robby, W. Yang, J. Su, and J. Liu, “Attentive generative adversarial network for raindrop removal from a single image,” in 2018 Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2018, pp. 2482–2491.
* EfficientDeRain (KPN): G. Qing, S. Jingyang, J. Felix, M. Lei, X. Xiaofei, F. Wei, and L. Yang, “Efficientderain: Learning pixel-wise dilation filtering for high-efficiency single-image deraining,” in 2021 AAAI Conference on Artificial Intelligence, February 2021. 
* SA-GAN: H. Zhang, I. Goodfellow, D. Metaxas, and A. Odena, “Self-attention generative adversarial networks,” arXiv preprint arXiv:1805.08318, January 2019.
* Dilated Convolutions: F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” International Conference on Learning Representations (ICLR), April 2016.
* JORDER: W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, “Deep Joint Rain Detection and Removal from a Single Image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), March 2017, pp. 1357–1366.
* DID-MDN: H. Zhang and V. M. Patel, “Density-aware single image de-raining using a multi-stream dense network,” in 2018 Proceedings of the IEEE Conference on Computer Vision and Pattern and Recognition (CVPR), June 2018, pp. 695–704.
* U-net transformer: O. Petit, N. Thome, C. Rambour, and L. Soler, “U-net transformer: Self and cross attention for medical image segmentation,” March 2021.
* SR-GAN: C. Ledig, L. Theis, F. Huszar, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, and W. Shi, “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network,” in Proceedings of the IEEE/ CVF Conference on Computer Vision and Pattern Recognition (CVPR), July 2017, pp. 4681–4690.
* ID-CGAN: H. Zhang, V. Sindagi, and V. M. Patel, “Image de-raining using a conditional generative adversarial network,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 11, pp. 3943–3956, November 2020.
* Rain in Driving (RID dataset): S. Li, I. B. Araujo, W. Ren, Z. Wang, E. K. Tokuda, R. H. Junior, R. Cesar-Junior, J. Zhang, X. Guo, and X. Cao, “Single image deraining: A comprehensive benchmark analysis,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2019, pp. 3838–3847
* Rain100H dataset: W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, “Deep Joint Rain Detection and Removal from a Single Image,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), March 2017, pp. 1357–1366.
* Rain1400 dataset: Rain100H: X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding, and J. Paisley, “Removing rain from single images via a deep detail network,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), July 2017, pp. 3855–3863