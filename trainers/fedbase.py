import tensorflow_federated as tff
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import time
import abc
import collections


class BaseClient(object):

    def __init__(self, client_id, train_data, test_data, batch_size, shuffle_buffer, prefetch_buffer):
        self.id = client_id
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.batch_size = batch_size
        self.train_data = self.dataset_preprocess(train_data)
        self.test_data = self.dataset_preprocess(test_data)

    @property
    def element_spec(self):
        return self.train_data.element_spec

    def dataset_preprocess(self, data):

        def batch_format_fn(element):
            """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
            return collections.OrderedDict(
                x=element['pixels'],
                y=tf.reshape(element['label'], [-1, 1]))
        # 处理顺序: 重复数据->
        return data.shuffle(self.shuffle_buffer).batch(
            self.batch_size).map(batch_format_fn).prefetch(self.prefetch_buffer)


class FedBase(abc.ABC):

    def __init__(self, params, model_params, model_creator, tff_dataset, optimizer, append2metric=None):
        """
        联邦学习框基类
        :param params: 参数
        :param learner: 需要学习的模型
        :param dataset: 数据集
        :param optimizer: 优化器, 这个用于创建静态图的 loss 的 op
        """
        # 显式指定参数
        self.optimizer = optimizer
        self.seed = params['seed']
        self.max_clients_num = params['max_clients_num']
        self.num_epochs = params['num_epochs']
        self.num_rounds = params['num_rounds']
        self.clients_per_round = params['clients_per_round']
        self.save_every_round = params['save_every']
        # 在 train 和 test
        self.eval_every_round = params['eval_every']
        self.batch_size = params['batch_size']
        self.hide_client_output = params['quiet']
        # 这个是整体的数据对象
        self.train_data, self.test_data = tff_dataset
        self.clients = self.setup_clients()
        #
        self.input_data_element_type = self.clients[0].element_spec
        self.fed_process = self.create_algorithms(params, model_params=model_params, model_creator=model_creator)
        self.fed_state = self.fed_process.initialize()
        # 定义 metric
        print('Input shape {}'.format(self.input_data_element_type))

    @property
    def num_clients(self):
        return len(self.clients)

    def setup_clients(self):
        clients = []
        for i in range(self.max_clients_num):
            cid = self.train_data.client_ids[i]
            train_dataset = self.train_data.create_tf_dataset_for_client(cid)
            test_dataset = self.test_data.create_tf_dataset_for_client(cid)
            clients.append(BaseClient(cid, train_data=train_dataset,
                                      test_data=test_dataset,
                                      shuffle_buffer=100,
                                      prefetch_buffer=10,
                                      batch_size=self.batch_size))
            print('Choose client:', cid)
        return clients

    def create_algorithms(self, params, model_params, model_creator):
        # 初始模型
        def create_model():
            # We _must_ create a new model here, and _not_ capture it from an external
            # scope. TFF will call this within different graph contexts.
            return model_creator(options=params, model_params=model_params, input_spec=self.input_data_element_type)
        fed_process = tff.learning.build_federated_averaging_process(
            create_model,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))
        print(str(fed_process.initialize.type_signature))
        return fed_process

    def select_clients(self, round_i):
        np.random.seed(round_i)
        indices = np.random.choice(range(self.num_clients), self.clients_per_round, replace=False)
        return indices, np.asarray(self.clients)[indices]

    def train(self):
        # state, metrics = self.fed_states.next(self.fed_states, federated_train_data)
        # print('round  1, metrics={}'.format(metrics))

        for round_num in range(self.num_rounds):
            client_id, clients = self.select_clients(round_num)
            state, metrics = self.fed_process.next(self.fed_state, [c.train_data for c in clients])
            print('round {:2d}, metrics={}'.format(round_num, metrics))
            self.fed_state = state