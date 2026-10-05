from setuptools import find_packages, setup

setup(
    name="credit_worthiness",
    version="0.1.0",
    author="Bartlomiej Boczylo",
    author_email="bartlomiej.boczylo@gmail.com",
    description="Production ready MLOps pipeline predicting the credit worthiness of loan applicants",
    packages=find_packages(),
    python_requires=">=3.12",
)
