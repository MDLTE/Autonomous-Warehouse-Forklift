from setuptools import find_packages, setup

package_name = 'puzzlebot_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Marcelo',
    maintainer_email='your_email@tec.mx',
    description='EKF Localization and Trajectory Control for PuzzleBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'goto_point = puzzlebot_ros.goto_point:main',
            'kalman = puzzlebot_ros.kalman:main',
        ],
    },
)
