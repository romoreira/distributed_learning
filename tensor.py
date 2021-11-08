import collections
import tensorflow as tf
import numpy as np
import tensorflow_federated as tff

print(tff.federated_computation(lambda: 'Hello, World!')())

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
print("emnist_train: " + str(emnist_train))
print("emnist_test: " + str(emnist_test))
print("eminist_train.client_ids: " + str(emnist_train.client_ids))

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

print("example_dataset: " + str(example_dataset))

print("Tamanho de client_ids: " + str(len(emnist_train.client_ids)))
print("element_type_structure: " + str(emnist_train.element_type_structure))

example_element = next(iter(example_dataset))

example_element['label'].numpy()

NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):
    def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER, seed=1).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(example_dataset)
print("Preprocessed_example_dataset: " + str(type(preprocessed_example_dataset)))

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))
print("sample_batch: " + str(type(sample_batch)))

print(sample_batch)


def make_federated_data(client_data, client_ids):
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
print("sample_clients: " + str(sample_clients))

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))