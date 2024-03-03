from setuptools import find_packages,setup
from typing import List

#Create a constant to remove '-e .'
HYPHEN_E_DOT='-e .'
def get_requirements(file_name:str)->List[str]:
    """
    This function returns the packages needed to install from requirements.txt
    """

    #Initialize the list to store the packages required
    requirements=[]
    with open(file_name) as file:
        requirements=file.readlines()
        #Remove newline from the result
        requirements=[req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
        return requirements


setup(
    name="MLProject",
    version="0.0.1",
    packages=find_packages(),
    author="Gayathri",
    author_email="gayathripvt284@gmail.com",
    install_requires=get_requirements("requirements.txt")
)