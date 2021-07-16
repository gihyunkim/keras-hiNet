from hiNet import HiNet
import glob
import cv2
import numpy as np
import os
from natsort import natsorted
import shutil


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_datasets():
    img_paths = natsorted(glob.glob(src_path+"*.png"))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_path = img_path.replace("\\","/")
        save_img_name = img_path.replace("_input", "").split("/")[-1]
        gt_img = cv2.imread(img_path.replace("input", "label"))
        resized_img = cv2.resize(img, (3072, 2560), cv2.INTER_CUBIC)
        all_parts_of_predicted_img = np.zeros((2560, 3072, 3), dtype=np.uint8)
        for j in range(10):
            for i in range(12):
                parts_img = resized_img[j*256:(j+1)*256, i*256:(i+1)*256, :]
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
                predicted_img2 = np.array(predicted_img2.squeeze(axis=0) * 255.0).astype(np.uint8)
                all_parts_of_predicted_img[j*256:(j+1)*256, i*256:(i+1)*256, :] = predicted_img2
        all_parts_of_predicted_img = cv2.resize(img, (3264, 2448), cv2.INTER_LINEAR)
        # cv2.imshow("test", all_parts_of_predicted_img)
        # cv2.waitKey(1)
        cv2.imwrite("./sample_submission/"+save_img_name,  all_parts_of_predicted_img)




def predict_denoised_img(inputs):
    model_outputs = model.predict(inputs)
    return model_outputs

if __name__ == "__main__":
    original_input_shape = (3264, 2448)
    input_shape=(256, 256 ,3)
    src_path = "./datasets/test_input_img/"
    weight_path = "./pretrained/hinet_00730.h5"
    hi = HiNet(input_shape)
    model = hi.hinet()
    model.load_weights(weight_path)
    load_datasets()