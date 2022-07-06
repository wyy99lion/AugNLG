""" Pytorch Distributed utils
    This piece of code was heavily inspired by the equivalent of Fairseq-py
    https://github.com/pytorch/fairseq
"""


from __future__ import print_function

import math
import pickle

import torch.distributed

from others.logging import logger


def is_master(gpu_ranks, device_id):
    return gpu_ranks[device_id] == 0


def multi_init(device_id, world_size,gpu_ranks):
    print(gpu_ranks)
    dist_init_method = 'tcp://localhost:10000'
    dist_world_size = world_size
    torch.distributed.init_process_group(
        backend='nccl', init_method=dist_init_method,
        world_size=dist_world_size, rank=gpu_ranks[device_id])
    '''
    distributed这个包在调用其他的方法之前，需要使用 torch.distributed.init_process_group() 函数进行初始化。这将阻止所有进程加入。
    初始化默认的分布式进程组，这也将初始化分布式程序包
    参数:
    backend (str or Backend) – 后端使用。根据构建时配置，有效值包括 mpi，gloo和nccl。该字段应该以小写字符串形式给出(例如"gloo")，也可以通过Backend访问属性(例如Backend.GLOO)。
    init_method (str, optional) – 指定如何初始化进程组的URL。
    world_size (int, optional) – 参与作业的进程数。
    rank (int, optional) – 当前流程的排名。
    timeout (timedelta, optional) – 针对进程组执行的操作超时，默认值等于30分钟，这仅适用于gloo后端。
    group_name (str, optional, deprecated) – 团队名字。
    '''
    gpu_rank = torch.distributed.get_rank()
    # 返回当前进程组的排名。Rank是分配给分布式进程组中每个进程的唯一标识符。它们总是从0到world_size的连续整数。
    if not is_master(gpu_ranks, device_id):
    #     print('not master')
        logger.disabled = True

    return gpu_rank



def all_reduce_and_rescale_tensors(tensors, rescale_denom,
                                   buffer_size=10485760):
    """All-reduce and rescale tensors in chunks of the specified size.在指定大小的块中全部减少和重新缩放张量。

    Args:
        tensors: list of Tensors to all-reduce要全部归约的张量列表
        rescale_denom: denominator for rescaling summed Tensors重新缩放总和张量的分母
        buffer_size: all-reduce chunk size in bytes全部减少块大小（以字节为单位）
        
    """
    # buffer size in bytes, determine equiv. # of elements based on data type
    # 缓冲区大小（以字节为单位），确定等值。 # 基于数据类型的元素
    buffer_t = tensors[0].new(
        math.ceil(buffer_size / tensors[0].element_size())).zero_()
    # tensors[0].new创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
    # math.ceil(x)返回大于等于参数x的最小整数,即对浮点数向上取整.
    # 返回单个元素的大小(以字节为单位)。
    # .zero_()用0填充tensor
    
    buffer = []

    def all_reduce_buffer():
        # copy tensors into buffer_t
        offset = 0
        for t in buffer:
            numel = t.numel() #返回数组t中元素的个数
            buffer_t[offset:offset+numel].copy_(t.view(-1))
            # 将元素从复制t到self张量并返回 self。
            # 将t复制到buffer_t的offset至offset+numel个元素
            offset += numel

        # all-reduce and rescale 全部减少和重新缩放
        torch.distributed.all_reduce(buffer_t[:offset])
        '''
        在所有机器上减少张量数据，通过获得最终的结果。在调用之后张量在所有过程中都是按位相同的。
        参数：
        tensor (Tensor  buffer_t[:offset])   – 输入和输出的集合。该函数在适当的位置运行。
        op (optional) – torch.distributed.ReduceOp枚举的价值之一。指定一个操作用来进行逐元素降低。
        group (ProcessGroup, optional) – 要处理的过程组
        async_op (bool, optional) – 这个op时候应该是一个异步操作
        返回值：
        如果async_op设置为真的话，是一个异步工作句柄。如果async_op为空或者部分组为空，就为None 。
        '''
        buffer_t.div_(rescale_denom)
        '''
        torch.div()方法将输入的每个元素除以一个常量，然后返回一个新的修改过的张量。
        参数
        inp:这是输入张量。
        other:这是一个要划分为输入inp的每个元素的数字。
        out:输出张量。
        返回：它返回张量。
        '''

        # copy all-reduced buffer back into tensors
        offset = 0
        for t in buffer:
            numel = t.numel()
            t.view(-1).copy_(buffer_t[offset:offset+numel])
            offset += numel

    filled = 0
    for t in tensors:
        sz = t.numel() * t.element_size()
        if sz > buffer_size:
            # tensor is bigger than buffer, all-reduce and rescale directly
            torch.distributed.all_reduce(t)
            t.div_(rescale_denom)
        elif filled + sz > buffer_size:
            # buffer is full, all-reduce and replace buffer with grad
            all_reduce_buffer()
            buffer = [t]
            filled = sz
        else:
            # add tensor to buffer
            buffer.append(t)
            filled += sz

    if len(buffer) > 0:
        all_reduce_buffer()


def all_gather_list(data, max_size=4096):
    """Gathers arbitrary data from all nodes into a list."""
    world_size = torch.distributed.get_world_size()
    if not hasattr(all_gather_list, '_in_buffer') or \
            max_size != all_gather_list._in_buffer.size():
        all_gather_list._in_buffer = torch.cuda.ByteTensor(max_size)
        all_gather_list._out_buffers = [
            torch.cuda.ByteTensor(max_size)
            for i in range(world_size)
        ]
    in_buffer = all_gather_list._in_buffer
    out_buffers = all_gather_list._out_buffers

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size < 255*256
    in_buffer[0] = enc_size // 255  # this encoding works for max_size < 65k
    in_buffer[1] = enc_size % 255
    in_buffer[2:enc_size+2] = torch.ByteTensor(list(enc))

    torch.distributed.all_gather(out_buffers, in_buffer.cuda())

    results = []
    for i in range(world_size):
        out_buffer = out_buffers[i]
        size = (255 * out_buffer[0].item()) + out_buffer[1].item()

        bytes_list = bytes(out_buffer[2:size+2].tolist())
        result = pickle.loads(bytes_list)
        results.append(result)
    return results
