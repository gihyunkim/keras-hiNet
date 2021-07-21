# keras-hiNet
Deep Learning Model which is called HiNet for Image Denoising 

Made for Dacon in Korea. 

# Environment
tensorflow-gpu == 1.13.2 
keras-radam == 0.15.0 
keras == 2.3.1 
albumentations == 1.0.3 

# train environment
Optimization : RAdam
loss : PSNR Loss
Augmentation : Albumentations(Rotate, Flip)
callbacks : Tensorboard, CosineAnnealing, ModelCheckPoint
learning rate : min(1e-7), max(2e-4)
