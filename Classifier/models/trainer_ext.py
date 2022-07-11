import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed
from models.reporter_ext import ReportMgr, Statistics
from others.logging import logger


def _tally_parameters(model):# 获得参数的总数量
    n_params = sum([p.nelement() for p in model.parameters()]) 
    # parameters()会返回一个生成器（迭代器）
    # nelement() 可以统计 tensor (张量) 的元素的个数。
    return n_params


def build_trainer(args, device_id, model, optim):

    grad_accum_count = args.accum_count #累积计数
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id]) #对应设备的gpu等级
    else:
        gpu_rank = 0
        n_gpu = 0
    print('gpu_rank %d' % gpu_rank)

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt") #记录日志，存在tensorboard_log_dir-Unmt
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    # print(tr)
    if (model):
        n_params = _tally_parameters(model)
        logger.info('* number of parameters: %d' % n_params) #参数数量

    return trainer


class Trainer(object):

    def __init__(self, args, model, optim,
                 grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 report_manager=None):
        # Basic attributes.基本属性
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps #保存检查点步骤
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count #grad累计数
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.将模型设置为训练模式
        if (model):
            self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):

        logger.info('Start training...')

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0  #累积
        normalization = 0
        train_iter = train_iter_fct() #训练迭代函数

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(distributed
                                                .all_gather_list
                                                (normalization))

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats) #梯度累积，下文定义

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats) #下文定义

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats


    def test(self, test_iter, step, cal_lead=False, cal_oracle=False):

        # Set model in validating mode.将模型设置为验证模式
        def _get_ngrams(n, text):
            # 返回由所有ngram构成的无序不重复元素集
            ngram_set = set() 
            # set() 函数创建一个无序不重复元素集,可进行关系测试,删除重复数据,还可以计算交集、差集、并集等。
            text_length = len(text)
            max_index_ngram_start = text_length - n 
            #最大索引 ngram 开始，即ngram的开始元素从1 至 max_index_ngram_start
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i:i + n]))
            return ngram_set

        def _block_tri(c, p): #tri块
            tri_c = _get_ngrams(3, c.split()) #获取c中所有的3gram
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    #如果交集>0
                    return True
            return False

        if (not cal_lead and not cal_oracle):
            self.model.eval() 
            # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化
        stats = Statistics()

        pred_path = self.args.result_path
        with open(pred_path, 'w') as save_pred:
            with torch.no_grad():
                for batch in test_iter:
                    # 测试迭代器中的batch
                    src = batch.src
                    segs = batch.segs
                    clss = batch.clss
                    mask = batch.mask_src
                    src_str = batch.src_str

                    sent_scores = self.model(src, segs, clss, mask)
                    batch_size = sent_scores.size(0)

                    for i in range(batch_size):
                        save_pred.write(str(float(sent_scores[i].cpu().data.numpy())) + '\t' + src_str[i] + '\n')
        return stats

    def _gradient_accumulation(self, true_batchs, normalization, total_stats, report_stats):
        # 梯度累积
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask_src

            # Run model
            sent_scores = self.model(src, segs, clss, mask)

            loss = self.loss(sent_scores, clss.float())
            loss = loss.sum()
            numel = loss.numel()
            (loss / numel).backward() #loss之和/元素个数

            #batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)
            batch_stats = Statistics(float(loss.cpu().data.numpy()*100), numel)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [p.grad.data for p in self.model.parameters()
                             if p.requires_grad
                             and p.grad is not None]
                    distributed.all_reduce_and_rescale_tensors(
                        grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        #在多步梯度累积的情况下，仅在累积批次后更新
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [p.grad.data for p in self.model.parameters()
                         if p.requires_grad
                         and p.grad is not None]
                distributed.all_reduce_and_rescale_tensors(
                    grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # state_dict 是一个简单的python的字典对象,将每一层与它的对应参数建立映射关系
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        #  os.path.join()函数用于路径拼接文件路径，可以传入多个路径 如果不存在以‘’/’开始的参数，则函数会自动加上
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        启动report管理器的简单功能
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases
        在多进程案例中收集统计信息

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)
                要收集的统计信息对象 或 None （在这种情况下它返回 None ）

        Returns:
            stat: the updated (or unchanged) stat object 更新（或未更改）的 stat 对象
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        report训练统计数据的简单功能（如果设置了 report_manager）请参阅文档的
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        报告统计信息的简单函数（如果设置了 report_manager）
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set如果设置了模型保存程序，则保存模型
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
