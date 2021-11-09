import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow_federated as tff
import tensorflow_datasets as tfds

np.set_printoptions(precision=4)

directory = "/home/rodrigo/PycharmProjects/distributed_learning/dataset/"

packets = tf.keras.preprocessing.image_dataset_from_directory(
    directory, labels='inferred', label_mode='int',
    class_names=None, color_mode='rgb', batch_size=32, image_size=(256,
    256), shuffle=True, seed=None, validation_split=None, subset=None,
    interpolation='bilinear', follow_links=False)
print("packets: "+str(packets))

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=20)
images, labels = next(img_gen.flow_from_directory(directory))
#print("images: "+str(images))
#print("labels: "+str(labels))


ds = tf.data.Dataset.from_generator(
    lambda: img_gen.flow_from_directory(directory),
    output_types=(tf.float32, tf.float32),
    output_shapes=([32,256,256,3], [32,4])
)

print("element_spec: "+str(ds.element_spec))
for images, label in ds.take(1):
    print('images.shape: ', images.shape)
    print('labels.shape: ', labels.shape)

print("ds: "+str(ds))

sampled_client_ids = [1, 2, 3, 4]

for client_id in sampled_client_ids[:5]:
    client_local_dataset = ds.create_tf_dataset_for_client(client_id)
    for example in client_local_dataset:
        print(example)