# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    #易读的时间格式、消息的级别名称(‘DEBUG’, ‘INFO’, ‘WARNING’, ‘ERROR’, ‘CRITICAL’)、记录的信息
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) ## Log等级总开关

    console_handler = logging.StreamHandler()
    #使用这个Handler可以向类似与sys.stdout或者sys.stderr的任何文件对象(file object)输出信息。
    #它的构造函数是： StreamHandler([strm])其中strm参数是一个文件对象。默认是sys.stderr
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
