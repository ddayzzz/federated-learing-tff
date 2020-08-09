import tensorflow_federated as tff
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Softmax, Dense, Flatten, Reshape


def create_cnn_model(model_params):
    """
    按照 ~/.keras.json 的定义, 使用 NHWC 的格式
    :param model_params:
    :return:
    """
    sz = model_params['image_size']
    num_class = model_params['num_class']
    return tf.keras.models.Sequential([
        Input(shape=(sz, sz)),
        Reshape(target_shape=[sz, sz, 1]),
        Conv2D(filters=32, kernel_size=5, padding='same', activation=tf.nn.relu),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=5, padding='same', activation=tf.nn.relu),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=2048, activation=tf.nn.relu),
        Dense(units=num_class, activation=None),
        Softmax()
    ])


def create_tff_model(options, model_params, input_spec):
    """
    工厂函数, 创建模型
    :param options: 传递的参数
    :param model_params: 模型的参数
    :param input_spec: 输入的格式, {key: TensorSpec}
    :return:
    """
    sz = model_params['image_size']
    model = options['model']
    if model == 'cnn':
        keras_model = create_cnn_model(model_params)
    else:
        raise ValueError('Not support {}'.format(model))
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

