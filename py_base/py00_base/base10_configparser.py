

import configparser

# https://docs.python.org/zh-cn/3/library/configparser.html
# python 的 configparser 库必会的方法
# 它最核心的几个方法是：
# ConfigParser()、write()、read()、get()、set()、sections()、items()、add_section()、
# getint()、getfloat()、getboolean()

"""
ini文件的基本结构

[section]
key = value

"""

# 定义parser
config = configparser.ConfigParser()
# 读取配置文件
cfgs = config.read('base10.ini')
print(cfgs)

# 获取配置
print(config.get('Common', 'home_dir'))
print(config.getint('Common', 'size'))
print(config.getboolean('Common', 'debug'))
print(config.getfloat('Frameworks', 'Python'))

# 获取所有section
sections = config.sections()
print("sections: ", sections)

# 判断某个section是否存在
print(config.has_section('Common'))
print(config.has_section('Common2'))

# 判断某个配置项是否存在
print(config.has_option('Common', "debug"))
print(config.has_option('Common', "debug2"))

# 获取某个section下的所有items
print(config.items('Common'))

# get的fallback，可以在配置不存在的时候获取默认值
d1 = config.get("Common", "debug2", fallback="true")
print(d1, type(d1))
d2 = config.getboolean("Common", "debug2", fallback="true")
d22 = bool(d2)
print(d2, type(d2))
print(d22, type(d22))

# 修改某个配置
print(config.get('Common', 'system_dir'))
config.set('Common', 'system_dir', '/data/exec')
print(config.get('Common', 'system_dir'))

# 有时使用set修改过配置后，可以将配置写会文件
# with open("config.ini", "w", encoding="utf-8") as f:
#     config.write(f)

# ini 的默认配置，只要在section=[DEFAULT]下的配置，在任何一个其他section都会被继承



