from setuptools import setup, find_packages

setup(
	name='ysaisample',
	version='1.1',
	author='Yuji Suehiro',
	packages=find_packages(),
	package_data={'': ['test-images-with-rotation.csv', 'haarcascade_eye.xml', 'haarcascade_frontalface_default.xml']},
	url='https://github.com/YujiSue/education/aisample',
	description='Codes for education.'
)
