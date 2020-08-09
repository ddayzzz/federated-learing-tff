import numpy as np
import importlib
import tensorflow as tf
import os
import random
# 复用读取数据的 api
from dataset.data_reader import get_tff_dataset
from config import DATASETS, TRAINERS_TONAMES, MODEL_PARAMS
from config import base_options, add_dynamic_options


def read_options():
    parser = base_options()
    parser = add_dynamic_options(parser)
    parsed = parser.parse_args()
    options = parsed.__dict__
    # 设置种子
    os.environ['PYTHONHASHSEED'] = str(options['seed'])
    np.random.seed(1 + options['seed'])
    tf.random.set_seed(12 + options['seed'])
    random.seed(123 + options['seed'])


    # 读取数据集
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    # 加载数据
    tff_dataset = get_tff_dataset(dataset_name, sub_data)

    # 加载模型的类
    model_path = '%s.%s' % ('models', dataset_name)
    mod = importlib.import_module(model_path)
    mod_creater = getattr(mod, 'create_tff_model')

    # 训练器
    trainer_path = 'trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS_TONAMES[options['algo']])

    # 定义对应网络的参数 dataset.model_name
    model_options = MODEL_PARAMS['.'.join((dataset_name, options['model']))]

    # 打印参数
    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> 参数:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, tff_dataset, model_options, mod_creater, trainer_class, dataset_name, sub_data


def main():
    # 数据的文件始终在其父目录
    dataset_prefix = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # 解析参数
    options, tff_dataset, model_options, mod_creater, trainer_class, dataset_name, sub_data = read_options()

    # train_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'train')
    # test_path = os.path.join(dataset_prefix, 'dataset', dataset_name, 'data', 'test')

    # 调用solver
    trainer = trainer_class(params=options, model_params=model_options,
                            model_creator=mod_creater, tff_dataset=tff_dataset, optimizer=None,
                            )
    trainer.train()


if __name__ == '__main__':
    main()
