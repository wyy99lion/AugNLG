""" Report manager utility """
from __future__ import print_function

import sys
import time
from datetime import datetime

from others.logging import logger


def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter 
        ## SummaryWriter压缩（包括）了所有内容
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S") #记录时间

        writer = SummaryWriter(tensorboard_log_dir,
                               comment="Unmt")
        ##创建writer object，log会被存入tensorboard_log_dir，后缀为"Unmt"
        #eg tensorboard_log_dir为'logging-1',则存入的文件为'logging-1-Unmt'
        
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class  Report管理器基类
    Inherited classes should override继承的类应该覆盖：:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            report_every(int)：每隔这么多句子报告状态
            start_time(float)：手动设置报表开始时间。 负值意味着您需要稍后设置或使用`start()`
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats, multigpu=False):
        """
        This is the user-defined batch-level traing progress
        report function.
        这是用户自定义的批次级训练进度报告功能。

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.旧的统计实例。
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:
            if multigpu:
                report_stats = \
                    Statistics.all_gather_stats(report_stats)
            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden要被覆盖 """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step报告步骤的统计信息

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard
        一个报告管理器，它在标准输出以及（可选）TensorBoard 上编写统计信息

        Args:
            report_every(int): Report status every this many sentences每隔多少句子报告状态
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None要使用的 TensorBoard 摘要编写器或无
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)
        #输出step, num_steps,learning_rate, self.start_time

        # Log the progress using the number of batches on the x-axis.
        # 使用 x 轴上的批次数记录进度。
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   self.progress_step)
        report_stats = Statistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train xent: %g' % train_stats.xent())

            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       step)

        if valid_stats is not None:
            self.log('Validation xent: %g at step %d' % (valid_stats.xent(), step))

            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step)


class Statistics(object):
    """
    Accumulator for loss statistics.用于损失统计的累加器
    Currently calculates:

    * accuracy
    * perplexity
    * elapsed time
    """

    def __init__(self, loss=0, n_docs=0, n_correct=0):
        self.loss = loss
        self.n_docs = n_docs
        self.start_time = time.time()

    @staticmethod
    def all_gather_stats(stat, max_size=4096):
        """
        Gather a `Statistics` object accross multiple process/nodes跨多个进程/节点收集“统计”对象

        Args:
            stat(:obj:Statistics): the statistics object to gather
                accross all processes/nodes
                收集所有进程/节点的统计信息对象
            max_size(int): max buffer size to use要使用的最大缓冲区大小

        Returns:
            `Statistics`, the update stats object更新统计对象
        """
        stats = Statistics.all_gather_stats_list([stat], max_size=max_size)
        return stats[0]

    @staticmethod
    def all_gather_stats_list(stat_list, max_size=4096):
        """
        Gather a `Statistics` list accross all processes/nodes
        收集所有进程/节点的“统计”列表

        Args:
            stat_list(list([`Statistics`])): list of statistics objects to
                gather accross all processes/nodes
                要在所有进程/节点中收集的统计对象列表
            max_size(int): max buffer size to use

        Returns:
            our_stats(list([`Statistics`])): list of updated stats
        """
        from torch.distributed import get_rank
        from distributed import all_gather_list

        # Get a list of world_size lists with len(stat_list) Statistics objects
        # 使用 len(stat_list) 统计对象获取 world_size 列表的列表
        all_stats = all_gather_list(stat_list, max_size=max_size)

        our_rank = get_rank()
        our_stats = all_stats[our_rank]
        for other_rank, stats in enumerate(all_stats):
            if other_rank == our_rank:
                continue
            for i, stat in enumerate(stats):
                our_stats[i].update(stat, update_n_src_words=True)
        return our_stats

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `Statistics` object
        通过将值与另一个 `Statistics` 对象相加来更新统计信息

        Args:
            stat: another statistic object
            update_n_src_words(bool): whether to update (sum) `n_src_words`
                or not

        """
        self.loss += stat.loss

        self.n_docs += stat.n_docs

    def xent(self):
        """ compute cross entropy """
        if (self.n_docs == 0):
            return 0
        return self.loss / self.n_docs

    def elapsed_time(self):
        """ compute elapsed time 计算经过的时间 """
        return time.time() - self.start_time

    def output(self, step, num_steps, learning_rate, start):
        """Write out statistics to stdout.将统计信息写入标准输出

        Args:
           step (int): current step
           n_batch (int): total batches
           start (int): start time of step.
        """
        t = self.elapsed_time()
        step_fmt = "%2d" % step
        if num_steps > 0:
            step_fmt = "%s/%5d" % (step_fmt, num_steps)
        logger.info(
            ("Step %s; xent: %4.2f; " +
             "lr: %7.7f; %3.0f docs/s; %6.0f sec")
            % (step_fmt,
               self.xent(),
               learning_rate,
               self.n_docs / (t + 1e-5),
               time.time() - start))
        sys.stdout.flush()

    def log_tensorboard(self, prefix, writer, learning_rate, step):
        """ display statistics to tensorboard """
        t = self.elapsed_time()
        writer.add_scalar(prefix + "/xent", self.xent(), step)
        writer.add_scalar(prefix + "/lr", learning_rate, step)
