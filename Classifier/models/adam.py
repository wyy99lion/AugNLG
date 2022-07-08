import math
import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups 可迭代参数以优化或定义参数组
        lr (float, optional): learning rate (default: 1e-3) 学习率
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
            # 可选参数，用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
            # 可选参数，添加到分母以提高数值稳定性的术语（默认值：1e-8）
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            # 可选参数，权重衰减（L2 惩罚）（默认值：0）
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
            #是否使用来自论文`On the Convergence of Adam and Beyond`_的该算法的AMSGrad变体（默认值：False）

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            # 可选参数，用于计算梯度及其平方的运行平均值的系数（默认值：（0.9，0.999））
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            # dict.setdefault(key, default=None)
            # 如果键不存在于字典中，将会添加该键并将default的值设为该键的默认值
            # 如果键存在于字典中，将读出该键原来对应的值，default的值不会覆盖原来已经存在的键的值。

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                # 重新评估模型并返回损失的closure。
        """
        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    # Adam 不支持稀疏梯度，请考虑使用 SparseAdam
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values梯度值的指数移动平均值
                    state['next_m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values平方梯度值的指数移动平均值
                    state['next_v'] = torch.zeros_like(p.data)

                next_m, next_v = state['next_m'], state['next_v'] #为0
                beta1, beta2 = group['betas'] #为0.9, 0.999

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                # 衰减一阶和二阶移动平均系数
                # 就地操作以同时更新平均值
                next_m.mul_(beta1).add_(1 - beta1, grad)  # 实现next_m=next_m·beta1+(1 - beta1, grad)
                # x.mul_(y)实现把x和y点对点相乘，其中x.mul_(y)是in-place操作，会把相乘的结果存储到x中。值得注意的是，x必须是tensor, y可以是tensor，也可以是数。
                # .add_()都能把两个张量加起来，但.add_是in-place操作，比如x.add_(y)，x+y的结果会存储到原来的x中。Torch里面所有带"_"的操作，都是in-place的。
                next_v.mul_(beta2).addcmul_(1 - beta2, grad, grad) #next_v=next_v·beta2+（（1 - beta2）*grad* grad）
                update = next_m / (next_v.sqrt() + group['eps']) 
                
                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # 只是将权重的平方添加到损失函数中*不是*使用 Adam 的 L2 正则化/权重衰减的正确方法，因为这会以奇怪的方式与 m 和 v 参数相互作用。
                # 相反，我们希望以不与 m/v 参数交互的方式衰减权重。 这相当于使用普通（非动量）SGD 将权重的平方与损失相加。
                if group['weight_decay'] > 0.0:
                    #权重衰减（L2 惩罚）> 0.0
                    update += group['weight_decay'] * p.data 
                    # p.data 类似于做了一个p的copy
                    #update = update + 权重衰减*可迭代参数

                lr_scheduled = group['lr']

                update_with_lr = lr_scheduled * update
                p.data.add_(-update_with_lr)

                state['step'] += 1

                # step_size = lr_scheduled * math.sqrt(bias_correction2) / bias_correction1
                # No bias correction
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']

        return loss
