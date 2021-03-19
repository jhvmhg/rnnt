from collections import defaultdict
import numpy as np

"""
参考
[CTC解码](https://blog.csdn.net/weixin_42615068/article/details/93767781)
[CTC宝藏](https://zhuanlan.zhihu.com/p/39266552)
[CTC详细介绍](https://distill.pub/2017/ctc/)
"""




ninf = float("-inf")


# 求每一列(即每个时刻)中最大值对应的softmax值
def softmax(logits):
    # 注意这里求e的次方时，次方数减去max_value其实不影响结果，因为最后可以化简成教科书上softmax的定义
    # 次方数加入减max_value是因为e的x次方与x的极限(x趋于无穷)为无穷，很容易溢出，所以为了计算时不溢出，就加入减max_value项
    # 次方数减去max_value后，e的该次方数总是在0到1范围内。
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def remove_blank(labels, blank=0):
    new_labels = []
    # 合并相同的标签
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    # 删除blank
    new_labels = [l for l in new_labels if l != blank]

    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


def greedy_decode(y, blank=0):
    # 按列取最大值，即每个时刻t上最大值对应的下标
    raw_rs = np.argmax(y, axis=1)
    # 移除blank,值为0的位置表示这个位置是blank
    rs = remove_blank(raw_rs, blank)
    return raw_rs, rs


def beam_decode(y, beam_size=10):
    # y是个二维数组，记录了所有时刻的所有项的概率
    T, V = y.shape
    # 将所有的y中值改为log是为了防止溢出，因为最后得到的p是y1..yn连乘，且yi都在0到1之间，可能会导致下溢出
    # 改成log(y)以后就变成连加了，这样就防止了下溢出
    log_y = np.log(y)
    # 初始的beam (prefix, scores)
    beam = [([], 0)]

    for t in range(T):
        new_beam = []
        for prefix, score in beam:
            for i in range(V):
                #                 print(prefix)
                #                 break
                new_prefix = prefix + [i]
                new_score = score + log_y[t][i]
                new_beam.append((new_prefix, new_score))
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]
    return beam


def _log_sum_exp(a, b):
    """
     np.log(np.exp(a) + np.exp(b))
    """

    if a < b:
        a, b = b, a
    if b == ninf:
        return a
    else:
        return a + np.log(1 + np.exp(b - a))


def log_sum_exp(*args):
    """
    from scipy.special import logsumexp
    logsumexp(args)
    """
    res = args[0]
    for e in args[1:]:
        res = _log_sum_exp(res, e)
    return res


def prefix_beam_decode(y, beam_size=5, blank=0):
    """
    对于 +i 的新前缀，分两种情况：
        1、生成新前缀new_prefix；
        2、更新合并旧前缀prefix
    从之前的prefix到new_prefix(prefix+i)，考虑prefix[-1]与i是否相同
    如果相同的话，直接拼接会抵消，得到的是prefix，而new_prefix需要需要中间有blank
    """

    T, V = y.shape
    log_y = np.log(y)
    # 最后一个字符是blank与最后一个字符为non-blank两种情况
    beam = [(tuple(), (0, ninf))]
    # 对于每一个时刻t
    for t in range(T):
        new_beam = defaultdict(lambda: (ninf, ninf))
        for prefix, (p_b, p_nb) in beam:
            for i in range(V):
                p = log_y[t, i]
                if i == blank:
                    new_p_b, new_p_nb = new_beam[prefix]
                    new_p_b = log_sum_exp(new_p_nb, p_b + p, p_nb + p)  # new_p_b: y+blk, per(y+blk+blk, y+blk)
                    new_beam[prefix] = (new_p_b, new_p_nb)
                else:
                    end_t = prefix[-1] if prefix else None

                    new_prefix = prefix + (i,)
                    new_p_b, new_p_nb = new_beam[new_prefix]

                    # 1、生成新前缀new_prefix
                    # 如果和前缀最后一个不相同，合并new_p_nb，以及加上前一时刻带blk的转移和不带blk的转移
                    if i != end_t:
                        new_p_nb = log_sum_exp(new_p_nb, p_b + p, p_nb + p)
                    # 如果和前缀最后一个相同，合并new_p_nb，以及加上前一时刻带blk的转移
                    else:
                        new_p_nb = log_sum_exp(new_p_nb, p_b + p)
                    new_beam[new_prefix] = (new_p_b, new_p_nb)

                    # 2、更新合并旧前缀
                    # 如果一样，对于得到prefix的旧前缀，合并就前缀以及加上前一时刻不带blk的概率
                    if i == end_t:
                        new_p_b, new_p_nb = new_beam[prefix]
                        new_p_nb = log_sum_exp(new_p_nb, p_nb + p)
                        new_beam[prefix] = (new_p_b, new_p_nb)
        beam = sorted(new_beam.items(), key=lambda x: log_sum_exp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    return beam
