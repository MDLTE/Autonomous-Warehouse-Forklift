import os
from glob import glob
from setuptools import setup

package_name = 'botoni_ui'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # web assets (index.html + roslib.min.js)
        (os.path.join('share', package_name, 'web'),
            glob('web/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Maje',
    maintainer_email='rodz@roboarena.local',
    description='Interfaz web del montacargas Botoni (Roboarena 4.0).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'web_server = botoni_ui.web_server:main',
            'ngrok_node = botoni_ui.ngrok_node:main'
        ],
    },
)
