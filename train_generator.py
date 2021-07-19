import keras
import os
import datetime
from generator import Denoise_Generator
from Utils.cyclical_learning_rate import CyclicLR
from Utils.Cosine_annealing import CosineAnnealingScheduler
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from hiNet import HiNet
from keras_radam import RAdam
from loss import *
from keras.utils.multi_gpu_utils import multi_gpu_model
import albumentations as alb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train():
    '''model configuration'''
    input_shape = (256, 256, 3)
    epochs = 10000
    batch_size = 8
    weight_decay = 1e-4
    lr = 1e-4
    optimizer = RAdam(learning_rate=lr)
    multi_gpu = False
    model_name = "hinet"

    '''Augmentation'''
    augs = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.Rotate(limit=30, p=0.5),
    ], additional_targets={'gt_image':'image'})

    '''call back'''
    log_dir = "./logs/%s/%s"%(model_name, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    weight_save_dir = "./save_weights/"
    if not os.path.isdir(weight_save_dir):
        os.mkdir(weight_save_dir)
    weight_save_file = "%s/%s_{epoch:05d}.h5"%(weight_save_dir, model_name)

    '''get datasets'''
    train_gen = Denoise_Generator("./datasets/", input_shape, batch_size, augs=augs, is_train=True)
    # valid_gen = Class_Generator("./datasets/val/", input_shape, class_num, batch_size, augs= [], is_train=False)

    step_size  = train_gen.__len__()
    step_size = step_size * 8
    if step_size < 622:
        step_size = 622
    print("Step size: ", step_size)

    '''train'''
    hi = HiNet(input_shape=input_shape, filters=96, weight_decay=weight_decay)
    model = hi.hinet()
    model.summary()
    if multi_gpu:
        model = multi_gpu_model(model, gpus=2)
    # model.load_weights("./save_weights/hinet_00730.h5")
    # model.save("./test.h5")
    model.compile(optimizer=optimizer, loss={"two_pred_and_input":psnr_loss})
    model.fit_generator(train_gen, use_multiprocessing=True,epochs=epochs,
                        max_queue_size=20, workers=8, initial_epoch=0,
                        callbacks=[TensorBoard(log_dir),
                                   # ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
                                   # CyclicLR(base_lr=1e-7, max_lr=2e-4, step_size=step_size*8, mode="triangular2"),
                                   CosineAnnealingScheduler(eta_min=1e-7, eta_max=2e-4),
                                   ModelCheckpoint(weight_save_file, monitor="loss", save_best_only=True)])

if __name__ == "__main__":
    train()