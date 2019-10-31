from distutils.core import setup

setup(
    name='GNEDNN_release',
    version='0.1',
    packages=['ops', 'cifar.utils', 'cifar.models', 'utils', 'imagenet.utils', 'imagenet.models'],
    url='',
    license='',
    author='ZhaodongChen',
    author_email='chenzd15thu@ucsb.edu',
    description='', requires=['torch', 'torchvision', 'tensorboardX', 'numpy', 'tqdm', 'scipy']
)
