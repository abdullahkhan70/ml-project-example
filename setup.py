from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."

def get_requirements(path: str) -> List[str]:
    """
    This function main function is that it will returns all,
    the packages which are mentioned in requirements.txt file.
    """
    requirements = []
    with open(path, "r") as file:
        requirements = file.readlines()
        # readlines() also returns "\n" symbol at that end of 
        # the line. So, we need to replace it with an empty space.
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name="ml-project-example",
    version="1.0.0",
    author="Abdullah Khan",
    author_email="abdullahkhan9003@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)