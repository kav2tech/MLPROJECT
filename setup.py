from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    Returns list of requirements from requirements.txt
    """
    requirements = []
    with open(file_path, encoding="utf-8") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='student_performance_prediction',  # ✅ no spaces
    version='0.1.0',
    author='Kavyansh Tyagi',
    author_email='kavyanshtyagi222@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='A package to predict student performance using machine learning',
)
