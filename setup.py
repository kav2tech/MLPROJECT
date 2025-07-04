from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    '''
     This function will return a list of requirements
    '''
    requirements= []
    with open(file_path) as file_obj:   
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        # removing -e . from requirements if present
        # -e . is used for editable installs, which is not needed in this case
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements 



setup(
    name='STUDENT PERFORMANCE PREDICTION',
    version='0.1',
    author='Kavyansh Tyagi',
    author_email= 'kavyanshtyagi222@gmail.com',
    packages = find_packages(),
    install_requires= get_requirements('requirements.txt'),
    description='A package to predict student performance using machine learning',  
)