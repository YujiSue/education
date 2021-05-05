from setuptools import setup
from setuptools import find_packages

setup(
	name='YSCodes4Education',
	version='0.1.0',
	author='Yuji Suehiro',
	packages=['test', 'test/test.py'],
	package_dir={'': 'test'},
	url='https://github.com/YujiSue/education',
	description='Sample codes used for education.',
)
