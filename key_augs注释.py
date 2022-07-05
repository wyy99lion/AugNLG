#coding=utf8

import sys, os
import argparse
import multiprocessing

REDDIT_PATH = ''
OUTPUT_DIR = ''

def grep(kws):
    k = '_'.join(kws.split())
    command = 'grep \"' + kws + '\" ' + REDDIT_PATH + ' > ' + OUTPUT_DIR + k + '.txt'
    os.system(command)
    # system函数可以将字符串转化成命令在服务器上运行；其原理是每一条system函数执行时，其会创建一个子进程在系统上执行命令行，子进程的执行结果无法影响主进程；

if __name__ == '__main__':

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument("-domain", default='restaurant', type=str)
    parser.add_argument("-delex_path", default='./reddit.delex', type=str)
    parser.add_argument("-keyword_path", default='./augmented_data/[DOMAIN]_system.kws', type=str)
    parser.add_argument("-retrieve_path", default='./augmented_data/[DOMAIN].aug/', type=str)
    parser.add_argument("-thread_num", default=20, type=int)
    args = parser.parse_args()

    REDDIT_PATH = args.delex_path
    OUTPUT_DIR = args.retrieve_path.replace('[DOMAIN]', args.domain)
    if not os.path.exists(OUTPUT_DIR):
        os.system('mkdir ' + OUTPUT_DIR)
    input_path = args.keyword_path.replace('[DOMAIN]', args.domain)

    pool = multiprocessing.Pool(processes=args.thread_num)
    kws_list = []

    for line in open(input_path):
        # 循环到kws_list=20时，用grep计算kws_list对应的格式的列表
        if len(kws_list) == args.thread_num:
            utterances = pool.map(grep, kws_list)
            del kws_list[:]
        kws_list.append(line.strip())

    if len(kws_list) > 0:
        #计算20的余数的kws_list对应的格式
        utterances = pool.map(grep, kws_list)
