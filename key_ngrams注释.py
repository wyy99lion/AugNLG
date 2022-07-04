#coding=utf8

import re
import os
import sys
import random
import argparse
from nltk import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
    

def preprocess(string, replace_num=True, tokenization=True):
    
    '''
    对string分词后，词语间用空格填充，小写化
    数字替换为 '11111'
    '''
    if tokenization:
        string = ' '.join(word_tokenize(string))
        
    string = string.lower()
    if replace_num:
        numbers = re.findall(r"\d+\.?\d*",string)
        number_length_dict = {}
        for item in numbers:
            if len(item) not in number_length_dict:
                number_length_dict[len(item)] = []
            number_length_dict[len(item)].append(item)
        for num_len in sorted(number_length_dict.items(), key = lambda d:d[0], reverse = True):
            for number in num_len[1]:
                string = string.replace(number, '[NUMBER]')
    # replace is for tfidf tool
    return string.replace('[NUMBER]', '11111')

def ngram_validation_check(ngram):
    # ngram 验证检查。如果ngram关键词为空，返回false
    for tok in ngram:
        if len(tok) <= 1:
            return False
    return True

def get_ngrams(string, n):
    #将string转换为ngram组成的ngrams
    ngrams = []
    flist = string.split()
    for i in range(0, len(flist)-n+1):
        if not ngram_validation_check(flist[i:i+n]):
            continue
            #如果i对应的ngram为空，即其中没有词，跳出循环
        ngram = ' '.join(flist[i:i+n]) #将i至i+n个词定义为i对应的ngram
        ngrams.append(ngram) #将符合条件的ngram添加到ngrams
    return ngrams

def stopword_filtered(stopwords, key):
    #过滤停用词，非停用词返回false
    for tok in key.split():
        if tok not in stopwords:
            return False
    return True

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
    parser.add_argument("-in_domain_path", default='./domains/[DOMAIN]/train.txt', type=str)
    parser.add_argument("-delex_path", default='./reddit.delex', type=str)
    parser.add_argument("-keyword_path", default='./augmented_data/[DOMAIN]_system.kws', type=str)
    parser.add_argument("-out_domain_example_num", default=636486504, type=int)
    parser.add_argument("-out_domain_sample_num", default=50000, type=int)
    parser.add_argument("-ngrams", default=2, type=int) #ngram=2
    # filtering thresholds过滤阈值
    parser.add_argument("-in_out_ratio", default=0.1, type=float)
    parser.add_argument("-min_count", default=1, type=int)
    parser.add_argument("-min_tf_idf", default=0.01, type=float)
    args = parser.parse_args()

    in_domain_path = args.in_domain_path
    delex_path = args.delex_path
    keyword_path = args.keyword_path
    out_domain_example_num = args.out_domain_example_num
    out_domain_sample_num = args.out_domain_sample_num

    # Get randomly sampled out of domain examples随机选取域外例子
    sample_ids = [random.randint(0, out_domain_example_num) for i in range(0, out_domain_sample_num)]
    out_domain_exps = []
    for i, line in enumerate(open(delex_path)):
        #取一定数量范围内的域外的例子
        #if i in sample_ids:
        #    out_domain_exps.append(line.strip())
        if i >= out_domain_sample_num:
            break
            #数量超过范围跳出循环
        out_domain_exps.append(line.strip())


    # Get in domain examples获得域内例子
    in_domain_exps = []
    in_domain_path = in_domain_path.replace('[DOMAIN]', args.domain)
    for line in open(in_domain_path):
        flist = line.strip().split(' & ')  #去除头尾空字符后，用&分割得到的字符串列表
        #strip()方法用于移除字符串头尾指定的字符(默认为空格)。
        #.split(' & ') 使用分隔符' & '分割
        if len(flist) != 2:
            continue
            #如果其line中字符串数量（句子数量）少于2，跳出循环
        utterance = flist[1].strip() #选取该行第二个句子，去除首位字符
        in_domain_exps.append(utterance) #将所有行的第二个句子添加到in_domain_exps


    # In-domain examples域内例子
    in_domain_exps = [preprocess(string, tokenization=False) for string in in_domain_exps]
    #拆分域内例子句子为字符串构成的列表，小写，去除数字
    in_domain_corpus = ' '.join(in_domain_exps) #转换为string
    # Ngram extractionNgram 提取
    in_domain_ngrams = []
    for string in in_domain_exps:
        in_domain_ngrams += get_ngrams(string, args.ngrams)
        #获得每个域内例句的ngram构成（ngrams），组合成为in_domain_ngrams
    in_count = Counter(in_domain_ngrams)
    #对字符串\列表\元祖\字典（域内ngram关键词）进行计数,返回一个字典类型的数据,键是元素,值是元素出现的次数


    # Out-domain utterances
    #out_domain_exps = [preprocess(string) for string in out_domain_exps]
    output_domain_corpus = ' '.join(out_domain_exps)
    #拆分域外例子句子为字符串构成的列表，小写，去除数字，转换为string


    # Get tf-idf
    corpus = [in_domain_corpus, output_domain_corpus] #域内域外句子构成的处理后得到的句子string
    vocabulary = {}
    for key in in_count:
        vocabulary[key] = len(vocabulary) #为每个关键词赋予编号，获得每个ngram关键词为键，对应编号为值的字典
    vectorizer = TfidfVectorizer(vocabulary=vocabulary, smooth_idf=True, use_idf=True, ngram_range=(2, 2))
    '''
    TfidfVectorizer： 除了考量某词汇在文本出现的频率，还关注包含这个词汇的所有文本的数量。
    能够削减高频没有意义的词汇出现带来的影响, 挖掘更有意义的特征。属于Tfidf特征。
    TF为某个词在文章中出现的总次数。为了消除不同文章大小之间的差异，便于不同文章之间的比较，我们在此标准化词频：TF = 某个词在文章中出现的总次数/文章的总词数。
    IDF为逆文档频率。逆文档频率（IDF） = log（词料库的文档总数/包含该词的文档数+1）。
    为了避免分母为0，所以在分母上加1。
    TF-IDF值 = TF * IDF。
    字典为vocabulary
    ngram_range：例如ngram_range(min,max)，是指将text分成min，min+1，min+2,.........max 个不同的词组。
            比如 '我 爱 中国' 中ngram_range(1,3)之后可得到'我'  '爱'  '中国'  '我 爱'  '爱 中国' 和'我 爱 中国'，如果是ngram_range (1,1) 则只能得到单个单词'我'  '爱'和'中国'。

    '''
    #vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=(2, 2))


    # Read result
    tfidf = vectorizer.fit_transform(corpus)  #文本矩阵,得到tf-idf矩阵，稀疏矩阵表示法（(n_samples, n_features) 的X稀疏矩阵）
    tfidf_tokens = vectorizer.get_feature_names()  #显示所有文本的词汇的列表
    df_tfidfvect = pd.DataFrame(data = tfidf.toarray(), index = ['In_domain', 'Out_domain'], columns = tfidf_tokens).to_dict()
    # tfidf.toarray() 是将结果转化为稀疏矩阵
    #in_domain_sorted = df_tfidfvect[:1].sort_values(by=['In_domain'], axis=1).to_dict()


    # First filtering 筛选出TFIDF分数符合要求的ngram短语
    new_dict = {}; stop_words = set()
    print ('Filted phrases: ')
    for key in df_tfidfvect:
        tf_idf = df_tfidfvect[key]['In_domain']
        #print (key, tf_idf, df_tfidfvect[key]['Out_domain'], in_count[key])
        if tf_idf * args.in_out_ratio < df_tfidfvect[key]['Out_domain']:  #df_tfidfvect[key]['Out_domain']>0.1*tf_idf 阈值
            # 域外ngram对应值 大于 域内ngram*0.1
            print (key, tf_idf, df_tfidfvect[key]['Out_domain'], in_count[key])
            #输出ngram对应的 域内值，域外值，出现的总次数
            for tok in key.split():
                stop_words.add(tok)
                #不符合条件的key中的tok呗加入停用词
            continue
        if in_count[key] >= args.min_count and tf_idf >= args.min_tf_idf:
            new_dict[key] = tf_idf
            #符合条件的域内ngram短语的值添加到new_dict


    # Second filtering 根据表面重叠关键词获取领域相关的话语
    #过滤掉带停用词的ngram
    filtered_ngrams = {}
    for key in new_dict:
        if stopword_filtered(stop_words, key):
            continue
        filtered_ngrams[key] = new_dict[key]
        

    # Write out
    fpout = open(keyword_path.replace('[DOMAIN]', args.domain), 'w')
    for item in sorted(filtered_ngrams.items(), key = lambda d:d[1], reverse=True):
        fpout.write(item[0] + '\n')
    fpout.close()
