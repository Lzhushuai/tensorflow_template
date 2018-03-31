"""
* Project Name: Tensorflow Template
* Author: huay
* Mail: imhuay@163.com
* Created Time:  2018-3-20 17:49:53
"""
from setuptools import setup, find_packages

install_requires = [
    'bunch',
    'huaytools'
    # 'tensorflow',  # install it beforehand
]

setup(
    name="tensorflow_template",
    version="1.0",
    keywords=("huay", "tensorflow", "template", "tensorflow template"),
    description="A tensorflow template for quick starting a deep learning project.",
    long_description="A deep learning template with tensorflow and it will help you "
                     "to change just the core part of model every time you start a new tensorflow project.",
    license="MIT Licence",
    url="https://github.com/imhuay/tensorflow_template",
    author="huay",
    author_email="imhuay@163.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="any",
    install_requires=install_requires
)
