from hiNet import HiNet
import glob
import cv2
import numpy as np
import os
from natsort import natsorted
import shutil
from tqdm import tqdm
from keras.utils.multi_gpu_utils import multi_gpu_model


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def predict():
    img_paths = natsorted(glob.glob(src_path+"*.png"))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_path = img_path.replace("\\","/")
        save_img_name = img_path.replace("_input", "").split("/")[-1]
        gt_img = cv2.imread(img_path.replace("input", "label"))
        resized_img = cv2.resize(img, (input_shape[0], input_shape[1]), cv2.INTER_CUBIC)
        model_input_img = resized_img / 255.0
        model_input_img = np.expand_dims(model_input_img, axis=0)
        model_outputs = predict_denoised_img(model_input_img)

        '''stage 1'''
        predicted_img1 = model_input_img + model_outputs[:,:,:,3:6]
        predicted_img1 = np.clip(predicted_img1, 0, 1)
        predicted_img1 = np.array(predicted_img1.squeeze(axis=0) * 255.0).astype(np.uint8)

        '''stage 2'''
        predicted_img2 = model_input_img + model_outputs[:,:,:,6:]
        predicted_img2 = np.clip(predicted_img2, 0, 1)
        predicted_img2 = np.array(predicted_img2.squeeze(axis=0) * 255.0).astype(np.uint8)

        restored_img = cv2.resize(predicted_img2, (3264, 2448), cv2.INTER_LINEAR)
        cv2.imshow("test", restored_img)
        cv2.waitKey(1)
        cv2.imwrite("./sample_submission/" + save_img_name, restored_img)

def predict_with_sliding_window():
    img_paths = natsorted(glob.glob(src_path + "*.png"))
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img_path = img_path.replace("\\", "/")
        save_img_name = img_path.replace("_input", "").split("/")[-1]
        resized_img = cv2.resize(img, fixed_input_shape, cv2.INTER_CUBIC)
        all_parts_of_predicted_img = np.zeros((fixed_input_shape[1], fixed_input_shape[0], 3), dtype=np.float)

        sliding_window_size_h = int(fixed_input_shape[1]/stride)-(int(patch_size_h/stride)-1)
        sliding_window_size_w = int(fixed_input_shape[0]/stride)-(int(patch_size_w/stride)-1)

        for j in range(sliding_window_size_h):
            for i in range(sliding_window_size_w):
                print("( %d / %d )"%(j*sliding_window_size_w+i, sliding_window_size_w*sliding_window_size_h))
                parts_img = resized_img[j*stride:j*stride+patch_size_h, i*stride:i*stride+patch_size_w, :]
                model_input_img = parts_img / 255.0
                model_input_img = np.expand_dims(model_input_img, axis=0)
                model_outputs = predict_denoised_img(model_input_img)

                '''stage 1'''
                predicted_img1 = model_input_img + model_outputs[:,:,:,3:6]
                predicted_img1 = np.clip(predicted_img1, 0, 1)
                predicted_img1 = np.array(predicted_img1.squeeze(axis=0) * 255.0).astype(np.uint8)

                '''stage 2'''
                predicted_img2 = model_input_img + model_outputs[:,:,:,6:]
                predicted_img2 = np.clip(predicted_img2, 0, 1)
                predicted_img2 = np.array(predicted_img2.squeeze(axis=0) * 255.0).astype(np.float)

                all_parts_of_predicted_img[j*stride:j*stride+patch_size_h, i*stride:i*stride+patch_size_w, :] += predicted_img2
                overlapped = all_parts_of_predicted_img[j*stride:j*stride+(patch_size_h-stride), i*stride:i*stride+(patch_size_w-stride),:] # overlapped twice so that add later

                if j!=0: # not for first row
                    all_parts_of_predicted_img[j*stride:j*stride+(patch_size_h-stride), i*stride:i*stride+patch_size_w, :] /=2
                if i!=0: # not for first col
                    all_parts_of_predicted_img[j*stride:j*stride+patch_size_h, i*stride:i*stride+(patch_size_w-stride), :] /=2
                if j != 0 and i != 0:
                    all_parts_of_predicted_img[j * stride:j * stride + (patch_size_h - stride), i * stride:i * stride + (patch_size_w - stride), :] += overlapped

        all_parts_of_predicted_img = cv2.resize(all_parts_of_predicted_img.astype(dtype=np.uint8), (original_input_shape[0], original_input_shape[1]), cv2.INTER_LINEAR)
        show_img = cv2.resize(all_parts_of_predicted_img, (1024, 720))
        cv2.imshow("test", show_img)
        cv2.waitKey(1)
        cv2.imwrite("./sample_submission/"+save_img_name,  all_parts_of_predicted_img)


def predict_denoised_img(inputs):
    model_outputs = model.predict(inputs)
    return model_outputs

if __name__ == "__main__":
    input_shape=(256, 256 ,3)
    original_input_shape = (3264, 2448) # original image size
    fixed_input_shape = (3072, 2560) # rescale to be multiple of input_shape
    patch_size_w, patch_size_h = 256, 256 # patch size, must be equal with input_shape
    stride = 32
    if patch_size_w % stride !=0 or patch_size_h % stride !=0:
        print("ovelap param must be able to evenly divide patch_size (ex. 32, 64, 128)")
        exit(-1)
    src_path = "./datasets/test_input_img/"
    weight_path = "./save_weights_aug/11epochs.h5"
    hi = HiNet(input_shape, 96)
    model = hi.hinet()
    # model = multi_gpu_model(model, gpus=2)
    model.load_weights(weight_path)
    # model = model.layers[-2]
    # model.save("single_hinet.h5")
    # exit(-1)
    predict_with_sliding_window()