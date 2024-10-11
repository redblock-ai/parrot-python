"""
Setup.py enables user to install parrot locally! :D
"""

from setuptools import setup, find_packages

setup(
    name="parrot",
    version="0.1",
    description="A benchmarking package for evaluating LLM responses",
    author="Harsha Vardhan Khurdula",
    author_email="harsha@redblock.ai",
    packages=find_packages(),  
    install_requires=[
        "pandas",
        "pytest",
        "langchain-core",
        "langchain-ollama",
        "qa_metrics",
        "langchain_community",
        "regex"
    ],
    python_requires=">=3.0",
)
