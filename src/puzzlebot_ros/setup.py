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
            'aruco_detector = puzzlebot_ros.aruco_detector:main',
            'aruco_detector_sim = puzzlebot_ros.aruco_detector_sim:main',
            'poseKalman = puzzlebot_ros.poseKalman:main',
            'poseKalmanSim = puzzlebot_ros.poseKalmanSim:main',
            'alignment_node = puzzlebot_ros.alignment_node:main',
            'mission_manager = puzzlebot_ros.mission_manager:main',
            'qr_alignment_node = puzzlebot_ros.qr_alignment_node:main',
            'enc_calib_node = puzzlebot_ros.enc_calib_node:main',
            'qr_detector = puzzlebot_ros.qr_detector:main',
            'door_align = puzzlebot_ros.door_align:main',
        ],
    },
)
