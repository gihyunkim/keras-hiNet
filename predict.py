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
        model_outputs = predict_denoised_img(model_input_img)
        predicted_img = np.clip(model_outputs[1], 0, 1)
        print(predicted_img)
        predicted_img = np.array(predicted_img.squeeze(axis=0) * 255.0).astype(np.uint8)

        cv2.imshow("before", cv2.resize(img,(512,512)))
        cv2.imshow("after", predicted_img)
        cv2.waitKey()


def predict_denoised_img(inputs):
    model_outputs = model.predict(inputs)
    return model_outputs

if __name__ == "__main__":
    input_shape=(256, 256 ,3)
    src_path = "./datasets/train_input_img/"
    weight_path = "./save_weights/hinet_00129.h5"
    hi = HiNet(input_shape)
    model = hi.hinet()
    model.load_weights(weight_path)
    load_datasets()