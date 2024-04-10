from setuptools import setup, find_packages

setup(
    name='GraphReasoning',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        # Any dependencies your package needs. For example:
        # 'numpy>=1.19.2',
        # 'pandas>=1.1.3',
        numpy,
        pytorch,
        networkx,
        matplotlib,
        pandas,
        transformers>4.39.3,
    ],
    description='A brief description of your package.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='URL to your GitHub repo',
    classifiers=[
        # Choose your license as you wish
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        # Specify the Python versions you support here. For example:
        # 'Programming Language :: Python :: 3.7',
        # 'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
)
