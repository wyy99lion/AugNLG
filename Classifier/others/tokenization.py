# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import unicodedata
from io import open

#from pytorch_transformers import cached_path
from transformers import cached_path

logger = logging.getLogger(__name__) #获取日志文件

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
}
VOCAB_NAME = 'vocab.txt'


def load_vocab(vocab_file):
    #返回以单词为key，单词出现的位置为值的字典
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict() 
    #是一种特殊字典，能够按照键的插入顺序保留键值对在字典的次序
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip() #用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
            
            vocab[token] = index
            index += 1
    return vocab


def whitespace_tokenize(text):
    """
    将文本去除前后空字符后，拆分成列表
    Runs basic whitespace cleaning and splitting on a peice of text.
    对一段文本运行基本的空白清理和拆分。
    """
    text = text.strip() 
    if not text:
        return []
    tokens = text.split() #split()方法将字符串拆分为列表。默认分隔符是任何空白字符。
    return tokens


class BertTokenizer(object):
    """
    Runs end-to-end tokenization: punctuation splitting + wordpiece
    运行端到端标记化：标点符号拆分 + wordpiece
    """

    def __init__(self, vocab_file, do_lower_case=True, max_len=None,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", 
                              "[unused0]", "[unused1]", "[unused2]", "[unused3]", 
                              "[unused4]", "[unused5]", "[unused6]")):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.do_lower_case = do_lower_case
        self.vocab = load_vocab(vocab_file) #将返回以单词为key，单词出现的位置为值的字典
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        #返回以单词位置为key，单词为值的列表
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                              never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab) 
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text, use_bert_basic_tokenizer=False):
        split_tokens = []
        if(use_bert_basic_tokenizer):
            pretokens = self.basic_tokenizer.tokenize(text)
        else:
            pretokens = list(enumerate(text.split()))

        for i,token in pretokens:
            # if(self.do_lower_case):
            #     token = token.lower()
            subtokens = self.wordpiece_tokenizer.tokenize(token)
            for sub_token in subtokens:
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a sequence of tokens into ids using the vocab.
        使用 vocab 将一系列标记转换为 id。
        """
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        # if len(ids) > self.max_len:
        #     raise ValueError(
        #         "Token indices sequence length is longer than the specified maximum "
        #         " sequence length for this BERT model ({} > {}). Running this"
        #         " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
        #     )
        return ids

    def convert_ids_to_tokens(self, ids):
        """
        Converts a sequence of ids in wordpiece tokens using the vocab.
        使用 vocab 转换 wordpiece 标记中的 id 序列。
        """
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        从预训练模型文件中实例化 PreTrainedBertModel。
        如果需要，下载并缓存预训练的模型文件。
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = pretrained_model_name_or_path
        if os.path.isdir(vocab_file): #判断路径是否为目录
            vocab_file = os.path.join(vocab_file, VOCAB_NAME) #把目录和文件名合成一个路径
        # redirect to the cache, if necessary如有必要，重定向到缓存
        try:
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    vocab_file))
            return None
        if resolved_vocab_file == vocab_file:
            logger.info("loading vocabulary file {}".format(vocab_file)) #获取日志文件
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            # 如果我们使用的是预训练模型，请确保tokenizer索引序列的时间不会长于位置嵌入的数量
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path] #512
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len) 
            #.get('max_len', int(1e12))  get() 方法返回具有指定键的项目值。
        # Instantiate tokenizer.实例化分词器。
        tokenizer = cls(resolved_vocab_file, *inputs, **kwargs)
        return tokenizer


class BasicTokenizer(object):
    """
    Runs basic tokenization (punctuation splitting, lower casing, etc.).
    运行基本标记化（标点符号拆分、小写等）。
    """

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BasicTokenizer.构造一个 BasicTokenizer

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        """Tokenizes a piece of text.标记一段文本。"""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        '''
        这是在 2018 年 11 月 1 日为多语言和中文模型添加的。 
        这也适用于现在的英文模型，但没关系，因为英文模型没有针对任何中文数据进行训练，
        并且通常没有任何中文数据（词汇表中有汉字，因为维基百科确实有 英文维基百科中的一些中文单词。）。
        
        '''
        text = self._tokenize_chinese_chars(text) # 下文定义
        orig_tokens = whitespace_tokenize(text) #将文本去除前后空字符后，拆分成单词组成的列表
        split_tokens = []
        for i,token in enumerate(orig_tokens):
            #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token) #函数下文定义
            # split_tokens.append(token)
            split_tokens.extend([(i,t) for t in self._run_split_on_punc(token)])
            # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）

        # output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return split_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text.从一段文本中去除重音符号"""
        text = unicodedata.normalize("NFD", text) #unicodedata.normalize 函数对字符进行标准化。
        output = []
        for char in text:
            cat = unicodedata.category(char)
            #unicodedata.category():把一个字符返回它在UNICODE里分类的类型
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output) #将output中的元素用 连接

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text.在一段文本上拆分标点符号"""
        if text in self.never_split:
            return [text]
        chars = list(text) #将元组转换为列表
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            '''
            句子中单词没遍历完前，每遇到一个标点，重启一行（一个list）
            '''
            char = chars[i]
            if _is_punctuation(char):#检查 `chars` 是否是标点符号。
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output] #把每个list中的字符组合在一起
    

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char) #返回每个char对应的十进制整数
            '''
            ord() 函数是 chr() 函数（对于8位的ASCII字符串）或 unichr() 函数（对于Unicode对象）的配对函数，
            它以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值，
            如果所给的 Unicode 字符超出了你的 Python 定义范围，则会引发一个 TypeError 的异常。
            '''
            if self._is_chinese_char(cp):#通过十进制整数判断是否为中文
                #是中文前后加空格 append()方法用于在列表末尾添加新的对象
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:#非中文直接加
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        '''
        这将“中文字符”定义为 CJK Unicode 块中的任何内容：
        #https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        请注意，尽管它的名称，CJK Unicode 块并不全是日文和韩文字符。 
        现代韩文字母表是一个不同的块，日语平假名和片假名也是如此。 
        这些字母用于书写以空格分隔的单词，因此不会像所有其他语言一样对它们进行特殊处理和处理。
        '''
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """
        Performs invalid character removal and whitespace cleanup on text.
        对文本执行无效字符删除和空白清理
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):#_is_control(char)检查 `char` 是否是控制字符
                continue
            if _is_whitespace(char):#检查 `chars` 是否是空白字符。
                output.append(" ")
            else:
                output.append(char)
        return "".join(output) #获得去除控制字符、空白字符等的str


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization.运行 WordPiece 标记化。"""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        将一段文本标记为其单词片段。这使用贪婪的最长匹配优先算法来使用给定的词汇表执行标记化。

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.
            单个标记或空格分隔的标记。 这应该已经通过了 `BasicTokenizer`。

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):#去除前后空字符后单词列表
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars): #理论上将chars拆分为多个存在于self.vocab中的最长的substr
                '''
                eg 我 特别喜欢 。。。
                '''
                end = len(chars) #字符长度
                cur_substr = None
                while start < end:#筛选出以start开始，长度最长的 存在于self.vocab中的substr 
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:#如果以start开始没有在vocab中的substr，跳出循环
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character.检查 `chars` 是否是空白字符。"""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character.检查 `chars` 是否是控制字符。"""
    # These are technically control characters but we count them as whitespace
    # characters.这些在技术上是控制字符，但我们将它们视为空白字符。
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character.检查 `chars` 是否是标点符号"""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
