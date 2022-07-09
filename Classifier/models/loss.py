"""
This file handles the details of the loss function during training.

This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
该文件处理训练期间损失函数的细节。
这包括：LossComputeBase 和标准 NMTLossCompute，以及分片损失计算。
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reporter import Statistics


def abs_loss(generator, symbols, vocab_size, device, train=True, label_smoothing=0.0):
    compute = NMTLossCompute(
        generator, symbols, vocab_size,
        label_smoothing=label_smoothing if train else 0.0)
    compute.to(device)
    return compute



class LossComputeBase(nn.Module):#父类
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations
    用于管理有效损失计算的类。 处理分片下一步预测和累积多重损失计算


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.
    用户可以通过创建这个子类来实现自己的损失计算策略。 用户需要实现 _compute_loss() 和 make_shard_state() 方法。

    Args:
        generator (:obj:`nn.Module`) :将解码器的输出映射到目标词汇表上的分布的模块。
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :表示目标输出的 torchtext 词汇对象
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"通过“sents”或“tokens”进行标准化
    """

    def __init__(self, generator, pad_id):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.padding_idx = pad_id #填充id



    def _make_shard_state(self, batch, output,  attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        为 shards() 制作分片状态字典以返回可迭代的分片以进行有效的损失计算。 
        子类必须定义这个方法来匹配它自己的 _compute_loss() 接口。
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            计算示例的范围，整批还是截断？
            attns: the attns dictionary returned from the model.从模型返回的 attns 字典。
        """
        return NotImplementedError #在父类中不定义具体内容，在子类中定义，如不定义即调用，抛出error

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model #解码器输出`[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) #类型为FloatTensor:
              dictionary of attention distributions注意力分布词典
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.utils.Statistics`: loss statistics
        """
        shard_state = self._make_shard_state(batch, output)
        _, batch_stats = self._compute_loss(batch, **shard_state)

        return batch_stats

    def sharded_compute_loss(self, batch, output,
                              shard_size,
                             normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.
        计算前向损失和反向传播。 计算是通过分片完成的，并且可以选择截断以提高内存效率。

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.
        通过在解码器输出序列中取一个范围来反向传播，还支持长序列的截断 BPTT。
        范围来自`(cur_trunc, cur_trunc + trunc_size)`

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.
        注意分片是一种精确的效率技巧，可以减轻生成缓冲区所需的内存。 
        截断是一种近似的效率技巧，可以减轻 RNN 缓冲区所需的内存。

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window 截断窗口的起始位置
          trunc_size (int) : length of truncation window 截断窗口的长度
          shard_size (int) : maximum number of examples in a shard 分片中的最大示例数
          normalization (int) : Loss is divided by this number 损失除以这个数字

        Returns:
            :obj:`onmt.utils.Statistics`: validation loss statistics验证损失统计

        """
        batch_stats = Statistics() # models.reporter
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
                          .masked_select(non_padding) \
                          .sum() \
                          .item()
        num_non_padding = non_padding.sum().item()
        return Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt[:,1:],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output)
        scores = self.generator(bottled_output)
        gtruth =target.contiguous().view(-1)

        loss = self.criterion(scores, gtruth)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? """
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
