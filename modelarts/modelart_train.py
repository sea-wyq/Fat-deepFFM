# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===========================================================================
"""train_criteo."""
import argparse
import os

import moxing as mox
import numpy as np
from mindspore import context, Model, Tensor
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import export, load_checkpoint
from src.callback import AUCCallBack
from src.callback import TimeMonitor, LossCallback
from src.config import ModelConfig
from src.dataset import get_mindrecord_dataset
from src.fat_deepffm import ModelBuilder
from src.metrics import AUCMetric


def obs_data2modelarts(args):
    """copy code to modelarts"""
    if not mox.file.exists(args.modelarts_data_dir):
        mox.file.make_dirs(args.modelarts_data_dir)
    if not mox.file.exists(args.modelarts_result_dir):
        mox.file.make_dirs(args.modelarts_result_dir)
    mox.file.copy_parallel(src_url=args.data_url,
                           dst_url=args.modelarts_data_dir)


def modelarts_result2obs(args):
    """
    Copy result data from modelarts to obs.
    """
    mox.file.copy(src_url=os.path.join(args.modelarts_result_dir, "Fat-DeepFFM-"+ str(args.epochs) +"_41322.ckpt"),
                  dst_url=os.path.join(args.obs_result_dir, "Fat-DeepFFM-"+ str(args.epochs) +"_41322.ckpt"))
    mox.file.copy(src_url=os.path.join(args.modelarts_result_dir, 'fat-deepffm.air'),
                  dst_url=os.path.join(args.obs_result_dir, 'fat-deepffm.air'))


def export_AIR(args):
    """export air."""
    config = ModelConfig()
    model_builder = ModelBuilder(config)
    _, network = model_builder.get_train_eval_net()
    network.set_train(False)
    files = os.listdir(args.modelarts_result_dir)
    print("===>>>Files:", files)
    model_path = os.path.join(args.modelarts_result_dir, "Fat-DeepFFM-"+ str(args.epochs) +"_41322.ckpt")
    print(model_path)
    load_checkpoint(model_path, net=network)
    batch_ids = Tensor(np.zeros([config.batch_size, config.cats_dim]).astype(np.int32))
    batch_wts = Tensor(np.zeros([config.batch_size, config.dense_dim]).astype(np.float32))
    labels = Tensor(np.zeros([config.batch_size, 1]).astype(np.float32))
    input_data = [batch_ids, batch_wts, labels]
    export(network, *input_data, file_name=args.modelarts_result_dir + "/fat-deepffm", file_format="AIR")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CTR Prediction')
    parser.add_argument('--eval_file_name', type=str, default="./auc.log",
                        help='Auc log file path. Default: "./auc.log"')
    parser.add_argument('--loss_file_name', type=str, default="./loss.log",
                        help='Loss log file path. Default: "./loss.log"')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=("Ascend", "GPU", "CPU"),
                        help="device target, support Ascend, GPU and CPU.")
    parser.add_argument('--epochs', type=int, default=3, 
                        help="epochs of training")
    parser.add_argument('--do_eval', type=int, default=0,
                        help="Whether side training changes verification.")
    parser.add_argument('--data_url', type=str, default="obs://fat-deepffm-mindspore/data/criteo/mindrecord/",
                        help=" the path of data")
    parser.add_argument('--modelarts_data_dir', type=str, default="/cache/dataset",
                        help='modelart input path')
    parser.add_argument('--modelarts_result_dir', type=str, default="/cache/result",
                        help='modelart result path')
    parser.add_argument('--obs_result_dir', type=str, default="obs://fat-deepffm-mindspore/output/",
                        help='obs result path')
    ARGS = parser.parse_args()

    set_seed(1)
    obs_data2modelarts(ARGS)
    model_config = ModelConfig()
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    print("rank_size", rank_size)
    device_id = 0
    context.set_context(mode=context.GRAPH_MODE, device_target=ARGS.device_target, device_id=device_id)
    if ARGS.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', '0'))
    if rank_size == 1 or ARGS.device_target == "CPU":
        rank_id = 0
    elif rank_size > 1:
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=rank_size,
                                          parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if ARGS.device_target == "Ascend":
            context.set_auto_parallel_context(all_reduce_fusion_config=[9, 11])
    print("load dataset...")
    ds_train = get_mindrecord_dataset(ARGS.modelarts_data_dir, train_mode=True, epochs= ARGS.epochs,
                                      batch_size=model_config.batch_size,
                                      rank_size=rank_size, rank_id=rank_id, line_per_sample=1000)
    train_net, test_net = ModelBuilder(model_config).get_train_eval_net()
    auc_metric = AUCMetric()
    model = Model(train_net, eval_network=test_net, metrics={"AUC": auc_metric})
    time_callback = TimeMonitor(data_size=ds_train.get_dataset_size())
    loss_callback = LossCallback(ARGS.loss_file_name)
    cb = [loss_callback, time_callback]
    if rank_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * ARGS.epochs,
                                     keep_checkpoint_max=model_config.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix='Fat-DeepFFM', directory=ARGS.modelarts_result_dir, config=config_ck)
        cb += [ckpoint_cb]
    if ARGS.do_eval and rank_id == 0:
        ds_test = get_mindrecord_dataset(ARGS.modelarts_data_dir, train_mode=False)
        eval_callback = AUCCallBack(model, ds_test, eval_file_path=ARGS.eval_file_name)
        cb.append(eval_callback)
    print("Training started...")
    model.train(ARGS.epochs, train_dataset=ds_train, callbacks=cb, dataset_sink_mode=True)
    export_AIR(ARGS)
    modelarts_result2obs(ARGS)
