from setuptools import setup


setup(
        name='sparsely_activated_networks',
        version='0.0.0',
        url='https://github.com/pbizopoulos/sparsely-activated-networks',
        license='MIT',
        author='Paschalis Bizopoulos',
        author_email='pbizopoulos@protonmail.com',
        description='Sparsely Activated Networks',
        install_requires=['wfdb'],
        py_modules=['main'],
        classifiers=[
            'Development Status :: 6 - Mature',
            'License :: OSI Approved :: MIT License',
            ],
        )
