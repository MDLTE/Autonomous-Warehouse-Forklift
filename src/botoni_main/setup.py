from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'botoni_main'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Botoni Team',
    maintainer_email='botoni@roboarena.mx',
    description='Nodo maestro FSM de Botoni',
    license='MIT',
    entry_points={
        'console_scripts': [
            'botoni_fsm = botoni_main.botoni_fsm_node:main',
        ],
    },
)