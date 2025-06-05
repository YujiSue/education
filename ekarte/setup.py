from setuptools import setup, find_packages

setup(
	name='ysekarte',
	version='1.1',
	author='Yuji Suehiro',
	packages=find_packages(),
  package_data={'': ['sample.json']},
	url='https://github.com/YujiSue/education/ysekarte',
	description='Sample codes used for education.'
)