from setuptools import setup, find_packages

setup(
    name="backtesting",
    version="0.1.0",
    description="A backtesting system for crypto trading strategies",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "ccxt>=2.0.0",
        # "pandas-ta>=0.3.0",  # Comment out problematic dependency
        "python-dateutil>=2.8.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.7",  # Lower Python requirement for better compatibility
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
)
