
import logging

# https://docs.python.org/zh-cn/3/library/logging.html
# python的logging库必会的方法


# 添加filename参数，默认只输出到文件
logging.basicConfig(
    filename='logfile.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.debug('debug message')
logging.info('info message')

from pathlib import Path

current_file = Path(__file__).name
current_file = Path(__file__).resolve().name

# 输出到文件和控制台
logger = logging.getLogger(current_file + __name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 文件输出器
file_handler = logging.FileHandler('logfile.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 控制台输出
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stream_handler)


logger.debug('debug message2')
logger.info('info message2')


