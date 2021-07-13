from hiNet import HiNet
import glob
import cv2
import numpy as np


def load_datasets():
    img_paths = glob.glob(src_path+"*.png")
    for img_path in img_paths:
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (input_shape[0], input_shape[1]))
        model_input_img = resized_img / 255.0
        model_input_img = np.expand_dims(model_input_img, axis=0)
        model_output = predict_denoised_img(model_input_img)
        print(model_output)
        cv2.imshow("before", img)
        cv2.waitKey()


def predict_denoised_img(inputs):
    model.predict(inputs)


if __name__ == "__main__":
    input_shape=(64, 64 ,3)
    src_path = "./datasets/train_input_img/"
    weight_path = "./save_weights/hinet_00011.h5"
    hi = HiNet(input_shape)
    model = hi.hinet()
    model.load_weights(weight_path)
    load_datasets()