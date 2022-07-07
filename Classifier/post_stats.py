import argparse
from os import path
from functools import reduce
import re

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def n_grams(tokens, n):
    # tuple是另一种有序的列表，也称为“ 元组 ”。tuple 和 list 非常类似，但是，tuple一旦创建完毕，就不能修改了。
    # 获得tokens中的ngram并返回
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in range(l) if i + n < l]


def has_repeat(elements):
    # 如果元素存在重复，返回true
    d = set(elements)
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    return len(d) < len(elements)

def cal_self_repeat(summary):
    ngram_repeats = {2: 0, 4: 0, 8: 0}
    sents = summary.split('<q>')
    for n in ngram_repeats.keys():
        # keys函数是Python的字典函数，它返回字典中的所有键所组成的一个可迭代序列。使用keys()可以获得字典中所有的键。
        # 即，for n in [2,4,8]
        # Respect sentence boundary
        grams = reduce(lambda x, y: x + y, [n_grams(sent.split(), n) for sent in sents], [])
        '''
        获得 sents 中每个sent中的ngram和n，将其相加后返回
        
        reduce(function,iterable,initial)
        第一个参数是函数function，reduce()只能接受一个带有两个参数的函数；
        第二个参数是iterable，即可迭代对象，可以是列表、字符串等序列；
        第三个参数为初始值，可选可不选，但前两个参数是必须的；
        reduce()的用法：
        reduce()函数将一个序列内的所有元素按照序列顺序依次传入func函数中，并将得到的值继续作为参数与下一个序列中的元素进行操作，一直重复到序列中无元素为止；
        注意到reduce()的参数函数有两个参数了吧，如果没有指定初始值的话，那么传入函数的就是序列的前两个值，如果指定了初始值，那么传入的就是初始值和序列的第一个值
        '''
        ngram_repeats[n] += has_repeat(grams)
        #计算每一组ngram和n的组合有没有重复
    return ngram_repeats

def cal_novel(summary, gold, source, summary_ngram_novel, gold_ngram_novel):
    summary = summary.replace('<q>',' ')
    summary = re.sub(r' +', ' ', summary).strip() #将summary中的' +'替换为' '
    gold = gold.replace('<q>',' ')
    gold = re.sub(r' +', ' ', gold).strip()
    source = source.replace(' ##','')
    source = source.replace('[CLS]',' ').replace('[SEP]',' ').replace('[PAD]',' ')
    source = re.sub(r' +', ' ', source).strip()


    for n in summary_ngram_novel.keys():
        summary_grams = set(n_grams(summary.split(), n)) #创建一个由ngram和n构成的元素集
        gold_grams = set(n_grams(gold.split(), n))
        source_grams = set(n_grams(source.split(), n))
        joint = summary_grams.intersection(source_grams) #得到summary_grams和source_grams的交集
        # python集合的intersection()方法：intersection() 方法的工作原理是：返回多个集合（集合的数量大于等于2）的交集，即新的集合包含了所有集合中所共有的元素。
        novel = summary_grams - joint #得到summary_grams中除去（summary_grams和source_grams的交集）后的部分
        summary_ngram_novel[n][0] += 1.0*len(novel)
        summary_ngram_novel[n][1] += len(summary_grams)
        summary_ngram_novel[n][2] += 1.0 * len(novel) / (len(summary.split()) + 1e-6)
        # summary_ngram_novel[n]行的列元素为由其原本的值分别+ novel的长度、summary_grams的长度、（novel的长度/summary中元素长度+1e-6）
        joint = gold_grams.intersection(source_grams)
        novel = gold_grams - joint
        gold_ngram_novel[n][0] += 1.0*len(novel)
        gold_ngram_novel[n][1] += len(gold_grams)
        gold_ngram_novel[n][2] += 1.0 * len(novel) / (len(gold.split()) + 1e-6)


def cal_repeat(args):
    candidate_lines = open(args.result_path+'.candidate').read().strip().split('\n')
    gold_lines = open(args.result_path+'.gold').read().strip().split('\n')
    src_lines = open(args.result_path+'.raw_src').read().strip().split('\n')
    lines = zip(candidate_lines,gold_lines,src_lines)

    summary_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}
    gold_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 4: [0, 0, 0]}

    for c,g,s in lines:
        # self_repeats = cal_self_repeat(c)
        cal_novel(c, g, s,summary_ngram_novel, gold_ngram_novel)
    print(summary_ngram_novel, gold_ngram_novel)

    for n in summary_ngram_novel.keys():
        # summary_ngram_novel[n] = summary_ngram_novel[n][2]/len(src_lines)
        # gold_ngram_novel[n] = gold_ngram_novel[n][2]/len(src_lines)
        summary_ngram_novel[n] = summary_ngram_novel[n][0]/summary_ngram_novel[n][1]
        gold_ngram_novel[n] = gold_ngram_novel[n][0]/gold_ngram_novel[n][1]
    print(summary_ngram_novel, gold_ngram_novel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='', type=str)
    parser.add_argument("-result_path", default='../../results/cnndm.0')


    args = parser.parse_args()
    eval(args.mode + '(args)')
