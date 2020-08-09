from trainers.fedbase import FedBase


class FedAvg(FedBase):

    def __init__(self, params, model_params, model_creator, tff_dataset, optimizer):
        """
        联邦学习框基类
        :param params: 参数
        :param learner: 需要学习的模型
        :param dataset: 数据集
        :param optimizer: 优化器, 这个用于创建静态图的 loss 的 op
        """
        super(FedAvg, self).__init__(params=params, model_creator=model_creator, model_params=model_params, tff_dataset=tff_dataset, optimizer=optimizer, append2metric='')
