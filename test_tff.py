import collections
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

tff.federated_computation(lambda: 'Hello, World!')()
# 加载 EMNIST 数据
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
print('客户端的数量: ', len(emnist_train.client_ids))
print('训练数据的结构类型', emnist_train.element_type_structure)
############ 数据的访问
example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = next(iter(example_dataset))
example_element['label'].numpy()

############ 显示相客户端的数据的分布
# Number of examples per layer for a sample of clients
# f = plt.figure(figsize=(12, 7))
# f.suptitle('Label Counts for a Sample of Clients')
# for i in range(6):
#     # 获取第i个客户端的数据
#     client_dataset = emnist_train.create_tf_dataset_for_client(
#         emnist_train.client_ids[i])
#     plot_data = collections.defaultdict(list)
#     for example in client_dataset:
#         # Append counts individually per label to make plots
#         # more colorful instead of one color per plot.
#         label = example['label'].numpy()
#         plot_data[label].append(label)
#     plt.subplot(2, 3, i + 1)
#     plt.title('Client {}'.format(i))
#     for j in range(10):
#         plt.hist(
#             plot_data[j],
#             density=False,
#             bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# plt.show()

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

    return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))


def make_federated_data(client_data, client_ids):
    """
    为指定的客户端创建转换后的数据
    :param client_data:
    :param client_ids:
    :return:
    """
    return [
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]

# 这里仅仅使用固定数量的客户端
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))

# 定义模型
def create_keras_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(10, kernel_initializer='zeros'),
        tf.keras.layers.Softmax(),
    ])

# 包装成 tff 的模型
def model_fn():
    # We _must_ create a new model here, and _not_ capture it from an external
    # scope. TFF will call this within different graph contexts.
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

print(str(iterative_process.initialize.type_signature))
state = iterative_process.initialize()

state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

NUM_ROUNDS = 11
for round_num in range(2, NUM_ROUNDS):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))

