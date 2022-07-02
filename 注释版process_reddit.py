#coding=utf8

import os
import re
import sys
import json
import argparse
import multiprocessing
from nltk import word_tokenize

MIN_LEN = 2
MAX_LEN = 40

def str2bool(v):
    '''
        传入v
        根据输入返回true、false或报错
    '''
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def one_file(path):
    '''
        输入文件路径
        获得文件中符合条件（长度的）的回复语句
    '''
    utterances = []  #语句
    for line in open(path):
        json_obj = json.loads(line.strip()) 
        #strip()方法用于移除字符串头尾指定的字符(默认为空格)。
        response = json_obj["response"].strip().replace('\n', ' ').replace('\r', ' ')
        #替换转义字符为空格
        toks = response.split() #将整个字符串作为列表的一个元素返回
        #split()方法是用来拆分字符串的，返回的数据类型是列表
        #当传入参数时，必须指定分割符。
        #当不传递参数时，此时将整个字符串作为列表的一个元素返回
        if len(toks) < MIN_LEN:
            continue
        if len(toks) > MAX_LEN:
            continue
        utterances.append(response) #在长度范围内的response添加到utterances
    return utterances

def read_raw_data(base_path, output_path, thread_num=28):
    '''
        输入base和output文件路径，线程数量
    '''

    path_list = [] #文件列表
    pool = multiprocessing.Pool(processes=thread_num)
    #在利用Python进行系统管理的时候，特别是同时操作多个文件目录，或者远程控制多台主机，并行操作可以节约大量的时间。
    #当被操作对象数目不大时，可以直接利用multiprocessing中的Process动态成生多个进程
    #但如果是上百个，上千个目标，手动的去限制进程数量却又太过繁琐，此时可以发挥进程池的功效。
    #Pool可以提供指定数量的进程，供用户调用，当有新的请求提交到pool中时，如果池还没有满，那么就会创建一个新的进程用来执行该请求。
    #但如果池中的进程数已经达到规定最大值，那么该请求就会等待，直到池中有进程结束，才会创建新的进程来它。
    #例子https://www.cnblogs.com/kaituorensheng/p/4445418.html
    
    fpout = open(output_path, 'w')
    # open() 函数用于创建或打开指定文件；w:只写 (若文件存在，会覆盖文件；反之，则创建新文件)
    utt_num = 0  #对话数量0

    for filename in os.listdir(base_path):  #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        '''
            文件数量不足进程数：path_list加新文件名
            到达进程数：多进程按文件更新语句数量并将语句按行写入fpout
        '''
        if not filename.startswith('train-'):
            continue
        if len(path_list) == thread_num:
            utterances = pool.map(one_file, path_list) #获得path_list符合长度条件的所有句子
            '''
                Pool.map(func,iterable) 将可调用对象func应用给iterable中的所有项，然后以列表的形式返回结果。
                通过将iterable划分为多块并将工作分派给工作进程执行这项操作。
            '''
            for utts in utterances:
                utt_num += len(utts)  #更新语句数量
                fpout.write('\n'.join(utts) + '\n')  #将语句按行写入fpout
            print ('Processed', utt_num, 'utterances...')
            del path_list[:]  #清空path_list文件列表
        path_list.append(os.path.join(base_path, filename))

    if len(path_list) > 0:
        '''
            如果循环结束path_list仍有文件，即文件总数量并不刚好是thread_num=28的整数倍
            将剩下文件中的语句写入fpout并更新文件数量
        '''
        utterances = pool.map(one_file, path_list)
        for utts in utterances:
            utt_num += len(utts)
            fpout.write('\n'.join(utts) + '\n')

    print ('Number of utterances:', utt_num)
    fpout.close() #关闭fpout文件

def preprocess_number(string):
    #只输入string参数
    replace_num=True
    tokenization=True
    lower = True
    raw_string = string
    if tokenization:
        string = ' '.join(word_tokenize(string))
        #对string分词，词语间用空格填充，连接成一个字符串
    if lower:
        string = string.lower()
        #小写化
    if replace_num:
        numbers = re.findall(r"\d+\.?\d*",string)
        # re.findall()函数是返回某种形式(比如String)中所有与pattern匹配的全部字符串,返回形式为数组。
        #\d+ 表示可以出现1次或是n次数字；\.? 表示可以“.”可以出现一次，也可以不出现；\d* 表示可以出现0次或是n次数字
        number_length_dict = {}
        for item in numbers:
            '''
            将每个长度的数字分别添加到字典中以对应长度为索引的列表中
            '''
            if len(item) not in number_length_dict:
                number_length_dict[len(item)] = []
            number_length_dict[len(item)].append(item)
            
        for num_len in sorted(number_length_dict.items(), key = lambda d:d[0], reverse = True):
        '''
        将数字替换为'[NUMBER]'
        '''
            for number in num_len[1]:
                string = string.replace(number, '[NUMBER]')
    # replace is for tfidf tool
    return string.replace('[NUMBER]', '11111'), raw_string
    #返回用'11111'替换左右数字后的string和最初的string

def delexicalize_number(input_path, output_path, thread_num=28):
    #去词化number

    pool = multiprocessing.Pool(processes=thread_num)
    fpout = open(output_path, 'w')
    utt_list = []

    for line in open(input_path):
         '''
            文件数量不足进程数：path_list加新文件名
            到达进程数：多进程将每个句子的前两个utts写入fpout
        '''
        if len(utt_list) == thread_num:
            utterances = pool.map(preprocess_number, utt_list)
            for utts in utterances:
                fpout.write(utts[0] + '\t' + utts[1] + '\n')
            del utt_list[:]
        utt_list.append(line.strip())

    if len(utt_list) > 0:
        utterances = pool.map(preprocess_number, utt_list)
        for utts in utterances:
            fpout.write(utts[0] + '\t' + utts[1] + '\n')

    fpout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", default='read_raw', type=str, choices=['read_raw', 'delexicalization'])
    parser.add_argument("-min_length", default=2, type=int)
    parser.add_argument("-max_length", default=40, type=int)
    parser.add_argument("-base_path", default='/scratch/xxu/reddit/')
    parser.add_argument("-utterance_path", default='/scratch/xxu/reddit.utterances')
    parser.add_argument("-delex_path", default='/scratch/xxu/reddit.delex')
    parser.add_argument("-thread_num", default=20, type=int)
    args = parser.parse_args()
'''
argparse是一个Python模块：命令行选项、参数和子命令解析器。
主要有三个步骤：
创建 ArgumentParser() 对象
调用 add_argument() 方法添加参数:
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。
    action - 当参数在命令行中出现时使用的动作基本类型。
    nargs - 命令行参数应当消耗的数目。
    const - 被一些 action 和 nargs 选择所需求的常数。
    default - 当参数未在命令行中出现时使用的值。
    type - 命令行参数应当被转换成的类型。
    choices - 可用的参数的容器。
    required - 此命令行选项是否可省略 （仅选项可用）。
    help - 一个此选项作用的简单描述。
    metavar - 在使用方法消息中使用的参数值示例。
    dest - 被添加到 parse_args() 所返回对象上的属性名。
使用 parse_args() 解析添加的参数
'''
    MIN_LEN = args.min_length
    MAX_LEN = args.max_length

    if args.mode == 'read_raw':
        read_raw_data(args.base_path, args.utterance_path, args.thread_num)
    if args.mode == 'delexicalization':
        delexicalize_number(args.utterance_path, args.delex_path, args.thread_num)

