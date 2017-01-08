import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
def get_pill(pill_images_path='/media/zengxiao/ent/pillcom/dataset/pill201fb'):

    images = os.listdir(pill_images_path)
    pill_images = []

    for each in images:
        image_path = os.path.join(pill_images_path,each)
        image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        image = cv2.resize(image,(28,28))
        pill_images.append(image)

    X = np.array(pill_images)

    X_train, X_test= train_test_split(X, test_size = 0.2, random_state = 0)

    return (X_train,X_test)
if __name__ == '__main__':

    images = get_pill()
