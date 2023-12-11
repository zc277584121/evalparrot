from setuptools import setup, find_packages

setup(name='evalparrot',
      version='0.0.6',
      description='parrot evaluation',
      url='https://github.com/zc277584121/evalparrot.git',
      author='zc277584121',
      author_email='277584121@qq.com',
      license='MIT',
      packages=find_packages(),
      python_requires='>=3',
      install_requires=[
          "ragas==0.0.17",
          "datasets",
          "requests",
          "langchain",
          "tqdm",
          "numpy",
      ],
      )
