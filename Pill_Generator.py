import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import MyConfig
# def get_pill(pill_images_path='/media/zengxiao/ent/pillcom/dataset/pill201fb'):
#
#     images = os.listdir(pill_images_path)
#     pill_images = []
#
#     for each in images:
#         image_path = os.path.join(pill_images_path,each)
#         image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
#         image = cv2.resize(image,(28,28))
#         pill_images.append(image)
#
#     X = np.array(pill_images)
#
#     X_train, X_test= train_test_split(X, test_size = 0.2, random_state = 0)
#
#     return (X_train,X_test)

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    file_content_0 = tf.read_file(input_queue[0])
    image_0 = tf.image.decode_jpeg(file_content_0)

    return image_0

def preprocess(img):
    image = tf.cast(img, tf.float32)
    image = tf.image.resize_images(image, [100, 100])
    image = image/255.
    image = tf.random_crop(image, [100,100,3])
    return image

def get_batch(pill_images_path='/media/zengxiao/ent/pillcom/dataset/pill201fb',batch_size=64):

    image_paths = os.listdir(pill_images_path)
    image_paths = [each for each in image_paths if each.find('#')<0]


    # Reads pfathes of images together with their labels
    X_train_list = [os.path.join(pill_images_path, each) for each in image_paths]


    X_train = tf.convert_to_tensor(X_train_list, dtype=tf.string)


    # Makes an input queue
    input_queue = tf.train.slice_input_producer([X_train])

    X_train = read_images_from_disk(input_queue)

    # Optional Preprocessing or Data Augmentation
    # tf.image implements most of the standard image augmentation
    X_train = preprocess(X_train)


    # Optional Image and Label Batching
    X_train = tf.train.batch([X_train],batch_size=batch_size, capacity=1024)

    return X_train


if __name__ == '__main__':

    X_train = get_batch(MyConfig.pill_images_path)
    print X_train
