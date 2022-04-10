from setuptools import find_packages, setup
from monte_carlo_demo import __version__

setup(
    name="monte_carlo_demo",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="",
    author=""
)
