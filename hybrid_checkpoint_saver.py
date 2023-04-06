""" Checkpoint Saver

Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.

Hacked together by / Copyright 2020 Ross Wightman
"""

import glob
import operator
import os
import logging
from typing import Any, Callable, Dict, Optional, Union

import torch

from timm.utils.model import unwrap_model, get_state_dict
from timm.models import clean_state_dict, load_state_dict, load_checkpoint, remap_state_dict

_logger = logging.getLogger(__name__)


class HybridCheckpointSaver:
    def __init__(
            self,
            model,
            global_model, disk_router, hybrid_router,
            optimizer,
            args=None,
            model_ema=None,
            amp_scaler=None,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            max_history=10,
            unwrap_fn=unwrap_model):

        # objects to save state_dicts of
        self.model = model
        self.optimizer = optimizer
        self.global_model = global_model
        self.disk_router = disk_router
        self.hybrid_router = hybrid_router
        self.args = args
        self.model_ema = model_ema
        self.amp_scaler = amp_scaler

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.max_history = max_history
        self.unwrap_fn = unwrap_fn
        assert self.max_history >= 1

    def save_checkpoint(self, epoch, metric=None):
        assert epoch >= 0
        tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
        last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
        self._save(tmp_save_path, epoch, metric)
        if os.path.exists(last_save_path):
            os.unlink(last_save_path)  # required for Windows support.
        os.rename(tmp_save_path, last_save_path)
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            os.link(last_save_path, save_path)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            checkpoints_str = "Current checkpoints:\n"
            for c in self.checkpoint_files:
                checkpoints_str += ' {}\n'.format(c)
            _logger.info(checkpoints_str)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                if os.path.exists(best_save_path):
                    os.unlink(best_save_path)
                os.link(last_save_path, best_save_path)

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'arch': type(self.model).__name__.lower(),
            'global_arch': type(self.global_model).__name__.lower(),
            'disk_router_arch': type(self.disk_router).__name__.lower(),
            'hybrid_router_arch': type(self.hybrid_router).__name__.lower(),
            'state_dict': get_state_dict(self.model, self.unwrap_fn),
            'global_state_dict': get_state_dict(self.global_model, self.unwrap_fn),
            'disk_router_state_dict': get_state_dict(self.disk_router, self.unwrap_fn),
            'hybrid_router_state_dict': get_state_dict(self.hybrid_router, self.unwrap_fn),
            'optimizer': self.optimizer.state_dict(),
            'version': 2,  # version < 2 increments epoch before save
        }
        if self.args is not None:
            save_state['arch'] = self.args.model
            save_state['global_arch'] = self.args.global_model
            save_state['disk_router_arch'] = self.args.disk_router
            save_state['hybrid_router_arch'] = self.args.hybrid_router
            save_state['args'] = self.args
        if self.amp_scaler is not None:
            save_state[self.amp_scaler.state_dict_key] = self.amp_scaler.state_dict()
        if self.model_ema is not None:
            save_state['state_dict_ema'] = get_state_dict(self.model_ema, self.unwrap_fn)
        #if self.global_model_ema is not None:
        #    save_state['global_state_dict_ema'] = get_state_dict(self.global_model_ema, self.unwrap_fn)
        #if self.routing_model_ema is not None:
        #    save_state['routing_state_dict_ema'] = get_state_dict(self.routing_model_ema, self.unwrap_fn)
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                _logger.debug("Cleaning checkpoint: {}".format(d))
                os.remove(d[0])
            except Exception as e:
                _logger.error("Exception '{}' while deleting checkpoint".format(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, epoch, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, epoch)
        if os.path.exists(self.last_recovery_file):
            try:
                _logger.debug("Cleaning recovery: {}".format(self.last_recovery_file))
                os.remove(self.last_recovery_file)
            except Exception as e:
                _logger.error("Exception '{}' while removing {}".format(e, self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        return files[0] if len(files) else ''


def load_one_model(checkpoint, model, state_dict_name, log_info):
    if model is not None and state_dict_name in checkpoint:
        if log_info:
            _logger.info('Restoring ' + state_dict_name + ' state from checkpoint...')
        state_dict = clean_state_dict(checkpoint[state_dict_name])
        model.load_state_dict(state_dict)

def resume_hybrid_checkpoint(
        model: torch.nn.Module,
        global_model: torch.nn.Module,
        disk_router: torch.nn.Module,
        hybrid_router: torch.nn.Module,
        checkpoint_path: str,
        optimizer: torch.optim.Optimizer = None,
        loss_scaler: Any = None,
        log_info: bool = True,
):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            state_dict = clean_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)

            load_one_model( checkpoint, global_model, 'global_state_dict', log_info )
            load_one_model( checkpoint, disk_router, 'disk_router_state_dict', log_info )
            load_one_model( checkpoint, hybrid_router, 'hybrid_router_state_dict', log_info )

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

