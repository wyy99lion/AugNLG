#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time

import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'rnn_size']


def train_multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn') 
    #使用multiprocessing.get_context(method)函数来设置上下文中的启动方式

    # Create a thread to listen for errors in the child processes.
    # 创建一个线程来监听子进程中的错误
    error_queue = mp.SimpleQueue() #SimpleQueue() 先进先出
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,device_id, error_queue,), daemon=True))
        '''
        process模块是一个创建进程的模块，借助这个模块，就可以完成进程的创建。
        Process（group=None, target=None, name=None, args=(), kwargs={}）
        '''
        1 group——参数未使用，值始终为None
        2 target——表示调用对象，即子进程要执行的任务
        3 args——表示调用对象的位置参数元组，args=(1,2,'egon',)
        4 kwargs——表示调用对象的字典,kwargs={'name':'egon','age':18}
        5 name——为子进程的名称
        '''
        '''
        procs[i].start() # .start()：启动进程，并调用该子进程中的obj.run()
        logger.info(" Starting process pid: %d  " % procs[i].pid) #.pid：进程的pid
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])
    #setattr（）函数允许我们设置对象属性值。args的'gpu_ranks'属性设置为[int(i) for i in args.gpu_ranks]

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))
        # put()函数：把数值型或字符型变量转为字符型变量


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process.
    侦听子进程中的异常并将回溯传播到父进程的类。
    """

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()  #启动线程，即让线程开始执行
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get() #吃掉缓冲区回车
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)
        #os.getpid()可获取当前进程id，返回值为int

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage) ## Load all tensors onto GPU 1
    # torch.load(f, map_location=None, map_location=None, pickle_module=pickle, **pickle_load_args)
    '''
    name 类似文件的对象(必须实现read()，:meth ' readline '，:meth ' tell '和:meth ' seek ')，或者是包含文件的字符串。
    map_location – 函数、torch.device或者字典指明如何重新映射存储位置。
    pickle_module – 用于unpickling元数据和对象的模块(必须匹配用于序列化文件的pickle_module)
    pickle_load_args – (仅适用于Python 3)传递给pickle_module.load()和pickle_module.Unpickler()的可选关键字参数，例如errors=…
    '''
    
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
            #setattr（）函数允许我们设置对象属性值。args的k属性设置为opt[k]
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()
    #在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。

    filename = args.mode
    test_iter = data_loader.Dataloader(args, load_dataset(args, filename, shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)


def train_ext(args, device_id):
    if (args.world_size > 1):
        train_multi_ext(args)
    else:
        train_single_ext(args, device_id)


def train_single_ext(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, 
                load_dataset(args, 'train', shuffle=True), 
                args.batch_size, device,
                shuffle=True, is_test=False)

    model = ExtSummarizer(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)
