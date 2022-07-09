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
        batch_stats = Statistics() # 来自models.reporter
        shard_state = self._make_shard_state(batch, output)
        for shard in shards(shard_state, shard_size):#定义的函数
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(float(normalization)).backward() #div主要是用来求分组后行或列的占比
            batch_stats.update(stats) #update()函数用于将两个字典合并操作，有相同的就覆盖

        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.该批次的统计数据。
        """
        pred = scores.max(1)[1] #(1)dim=1表示每行的最大值；[1]表示返回最大值的索引
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
    标签平滑损失,降低onehot过拟合
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
        #（返回一个新字符串，表示将原字符串重复n次。size(0)中的0表示第0维度的数据数量）
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        '''
        scatter_(input, dim, index, src)：将src中数据根据index中的索引按照dim的方向填进input。可以理解成放置元素或者修改元素     

        dim：沿着哪个维度进行索引
        index：用来 scatter 的元素索引
        src：用来 scatter 的源元素，可以是一个标量或一个张量
        '''
        model_prob.masked_fill_((target == self.padding_idx).unsqueeze(1), 0)  #.unsqueeze(1)给张量升维
        #用value0填充tensor  model_prob中与mask(target == self.padding_idx).unsqueeze(1)中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。

        return F.kl_div(output, model_prob, reduction='sum') 
        # 计算 kl散度 F.kl_div()
        '''
        第一个参数传入的是一个对数概率矩阵，第二个参数传入的是概率矩阵。这里很重要，不然求出来的kl散度可能是个负值。
        有两个矩阵X, Y。因为kl散度具有不对称性，存在一个指导和被指导的关系，因此这连个矩阵输入的顺序需要确定一下。
        举个例子：如果现在想用Y指导X，第一个参数要传X，第二个要传Y。就是被指导的放在前面，然后求相应的概率和对数概率就可以了。
        '''


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, symbols, vocab_size,
                 label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, symbols['PAD'])
        self.sparse = not isinstance(generator[1], nn.LogSoftmax)
        # 来判断一个对象是否是一个已知的类型，通常用于判断两个类型是否相同。 
        if label_smoothing > 0:
            self.criterion = LabelSmoothingLoss(
                label_smoothing, vocab_size, ignore_index=self.padding_idx
            )
        else:
            self.criterion = nn.NLLLoss(
                ignore_index=self.padding_idx, reduction='sum'
            )
            #NLLLoss结果就是把输出结果过一层softmax后，取对数之后的结果与Label对应的那个值拿出来，再去掉负号，然后求和取均值。

    def _make_shard_state(self, batch, output):
        return {
            "output": output,
            "target": batch.tgt[:,1:],
        }

    def _compute_loss(self, batch, output, target):
        bottled_output = self._bottle(output) #output转换为dim1为output的dim2 的二维数据
        scores = self.generator(bottled_output)
        gtruth =target.contiguous().view(-1)
        # torch.contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列。
        # 先将不连续数据按语义转换为连续数据，后展平

        loss = self.criterion(scores, gtruth)

        stats = self._stats(loss.clone(), scores, gtruth)

        return loss, stats


def filter_shard_state(state, shard_size=None):
    """ ? ？？"""
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    # torch.split()作用将tensor分成块结构。把v拆分为shard_size的块
                    #参数：torch.split(tensor, split_size_or_sections, dim=0)
                    # tesnor：input，待分输入
                    # split_size_or_sections：需要切分的大小(int or list )
                    # dim：切分维度
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
        state：对应于 *LossCompute._make_shard_state() 输出的字典。 这些键的值类似于张量或无。
        shard_size：模型产生的分片的最大大小。
        eval_only：如果为真，只产生状态，没有别的。 否则，产生分片。

    Yields:
        Each yielded shard is a dict.每个产生的分片都是一个字典。

    Side effect:
        After the last shard, this function does back-propagation.
        在最后一个分片之后，此函数进行反向传播。
    """
    if eval_only:
        yield filter_shard_state(state)
        # key和切块后的数据
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        '''
        state 是一个类张量序列的字典，但我们想要一个张量字典序列。
        首先，将字典解压成一个键序列和一个类似张量的序列。
        '''
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))
        # 获得封装好的块，由键序列和一个类似张量的序列组成

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        '''
        现在，为每个分片生成一个字典。 钥匙总是一样的。 
        values 是一个长度为#keys 的序列，其中每个元素都是一个长度为#shards 的序列。 
        我们想要遍历分片，而不是键：因此，值需要通过分片重新压缩，然后每个分片都可以与键配对。
        '''
        for shard_tensors in zip(*values):
            #该函数的运算顺序：先对 * 后面的序列进行解包，之后用zip()函数进行打包。
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)
