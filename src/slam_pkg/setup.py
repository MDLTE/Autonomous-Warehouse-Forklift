import os
from glob import glob
from setuptools import setup

package_name = 'slam_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world')),
        *[
            (os.path.join('share', package_name, os.path.dirname(f)), [f])
            for f in glob('models/**', recursive=True)
            if os.path.isfile(f)
        ],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='patricio',
    maintainer_email='patricio@email.com',
    description='SLAM package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mapping_node = slam_pkg.mapping_node:main',
            'slam_node = slam_pkg.slam_node:main',
        ],
    },
)