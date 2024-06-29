from setuptools import find_packages, setup

setup(
   name='expansions',
   version='0.0.1',
   author='Orion Weller',
   author_email='oweller@cs.jhu.edu',
   packages=find_packages(),
   url='https://github.com/orionw/LM-expansions',
   description='Repository for testing LM-based expansions for IR.',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   install_requires=[
      'torch==1.13.1',
      'pyserini>=0.19',
      'transformers>=4.0',
      'pyyaml>=6.0',
      'ir_datasets>=0.5',
      'ftfy>=6.0',
      'accelerate>=0.15',
   ],
)
