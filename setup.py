from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow', 'keras==2.1.2', 'h5py','numpy']
DEP_LINK = ['http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl']

setup(
    name='s2s',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    dependency_links = DEP_LINK,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
