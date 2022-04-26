資料夾說明
-rainy_image_dataset: Dataset
-SensingGAN/SensingNet: 主要GAN與NET的架構和train、test
-models: 所有model
-data: PSNR、SSIM結果
-results: 除雨結果
-samples: 訓練過程時儲存的除雨結果，可以先看訓練過程變化來確認設定是否有誤
=====================================================================
SensingGAN/SensingNet/EfficientDerain:
  -使用Jupyter
  -檔案說明
    -主要model架構: SensingGAN/SensingNet主要差異為Net為Discriminator
      -network.py: SensingGAN模型架構
      -loss_function: SA-Feature Loss架構
        -VGG: from torchvision.models.vgg import vgg16
        -SA單元的channel: self.FeatureSA = Self_Attn(512, 'relu')
        -擷取的VGG層數: self.layer_name_mapping
        -SA應用的VGG層數
          if name in self.layer_name_mapping:
            if name == "relu5_3":
               x=self.FeatureSA(x)
      -dataset.py: 讀取Dataset並處理
      -spectral.py: Spectral Normalization
      -trainer.py: 訓練設定
         -if (epoch==9 or epoch==(opt.epochs-1)) and (i%20==0): 設定訓練時多久存一次除雨結果在samples資料夾
         -generator_loss: 各個loss的scale設定
      -utils.py: 共用funciton
         -get_files: 不同dataset的圖片讀取方式

    -SSIM計算:
      -pytorch_ssim資料夾

  -Train
    -train_own.ipynb: 調整以下參數以及架構後按restart重新執行即會開始訓練
      -os.environ["CUDA_VISIBLE_DEVICES"]: 指定GPU
      -save_path: model儲存位置
      -sample_path: 訓練過程影像儲存位置
      -baseroot: 讀取Training Data位置
      -train_batch_size: 訓練Batch Size
      -save_by_epoch: model每幾次Epoch儲存一次
      -epochs: 訓練Epochs
      -lr: Learning Rate
      -b1: Adam的Beta1
      -b2: Adam的Beta2
      
      -network.py: 需確認為想要的model架構
      -loss_function.py: 需確認為想要的loss function架構
      -trainer.py: 確認generator_loss各個loss的scale設定
      -utils.py: 確認get_files讀取的dataset

  -Test:
    -test.ipynb: 調整以下參數後以及架構按start執行即會開始測試
      -os.environ["CUDA_VISIBLE_DEVICES"]: 指定GPU
      -save_name: 儲存除雨圖位置
      -load_gname: 讀取model位置
      -baseroot: 讀取Testing Data位置
      -resize: 是否原比例調整圖片長寬
      -scale_size: 圖片長與寬最大值

      -network.py: 需更改為與load_gname model對應的架構
      -utils.py: 確認get_files讀取的dataset