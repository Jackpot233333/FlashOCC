import argparse
import sys
import os
sys.path.insert(0, os.getcwd())

import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

import os
from typing import Dict, Optional, Sequence, Union

import h5py
import mmcv
import numpy as np
import onnx
import pycuda.driver as cuda
import tensorrt as trt
import torch
import tqdm
from mmcv.runner import load_checkpoint
from mmdeploy.apis.core import no_mp
from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
from mmdeploy.utils import load_config
from packaging import version
from torch.utils.data import DataLoader

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module



def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument('--calib_num', type=int, help='num to calib')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model_prefix = args.prefix
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    import importlib
    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
    _module_dir = _module_dir.split('/')
    _module_path = _module_dir[0]

    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    print(_module_path)
    plg_lib = importlib.import_module(_module_path)

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    # assert model.img_view_transformer.grid_size[0] == 128
    # assert model.img_view_transformer.grid_size[1] == 128
    # assert model.img_view_transformer.grid_size[2] == 1
    if os.path.exists(args.checkpoint):
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    else:
        print(args.checkpoint, " does not exists!")
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    assert model.__class__.__name__ in ['BEVDetOCCAXMAXN']
    for i, data in enumerate(data_loader):
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        img = inputs[0]
        metas = model.get_bev_pool_input(inputs)
        if img.shape[0] > 6:
            img = img[:6]

        onnx_input = (img.float().contiguous(), metas[0].int().contiguous(),
            metas[1].int().contiguous())
        dynamic_axes = None
        input_names = [
                'img', 'indices_depth', 'indices_feat'
            ]

        with torch.no_grad():
            # if (model.wdet3d == True) and (model.wocc == False) :
            #     output_names=[f'output_{j}' for j in range(6 * len(model.pts_bbox_head.task_heads))]
            # elif (model.wdet3d == True) and (model.wocc == True) :
            #     output_names=[f'output_{j}' for j in range(1 + 6 * len(model.pts_bbox_head.task_heads))]
            # elif (model.wdet3d == False) and (model.wocc == True) :
            #     output_names=[f'output_{j}' for j in range(1)]
            # else:
            #     raise(" At least one of wdet3d and wocc is set as True!! ")

            # model.forward = model.forward_ori
            # torch.onnx.export(
            #     model,
            #     onnx_input,
            #     "./outputs/" + model_prefix + '.onnx',
            #     opset_version=13,
            #     dynamic_axes=dynamic_axes,
            #     input_names=input_names,
            #     output_names=output_names)
            # print('output_names:', output_names)
            # print('====== onnx is saved at : ', "./outputs/" + model_prefix + '.onnx')
            # # check onnx model
            # onnx_model = onnx.load("./outputs/" + model_prefix + '.onnx')
            # try:
            #     onnx.checker.check_model(onnx_model)
            # except Exception:
            #     print('ONNX Model Incorrect')
            # else:
            #     print('ONNX Model Correct')

            model.forward = model.forward_with_argmax
            output_names = [f'cls_occ_label']
            torch.onnx.export(
                model,
                onnx_input,
                "./outputs/" + model_prefix + '_maxn_with_argmax.onnx',
                opset_version=13,
                dynamic_axes=dynamic_axes,
                input_names=input_names,
                output_names=output_names)
            print('output_names:', output_names)
            print('====== onnx is saved at : ', "./outputs/" + model_prefix + '_maxn_with_argmax.onnx')
            # check onnx model
            onnx_model = onnx.load("./outputs/" + model_prefix + '_maxn_with_argmax.onnx')
            try:
                onnx.checker.check_model(onnx_model)
            except Exception:
                print('ONNX Model Incorrect')
            else:
                print('ONNX Model Correct')

        break


if __name__ == '__main__':
    # python3 tools/convert_bevdet_to_AXMAXN.py projects/configs/flashocc/flashocc-r50-M0-axmaxn.py ckpts/maxn.pth --fuse-conv-bn
    main()
