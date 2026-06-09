from setuptools import find_packages, setup

package_name = 'nav_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='patricio',
    maintainer_email='patricio@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'nav_node = nav_pkg.nav_node:main',
            'nav_node_sim = nav_pkg.nav_node_sim:main',
            'velocity_tester = nav_pkg.vel_test:main',
            'lidar_visualize = nav_pkg.lidar_visualize:main',
            'robot_viz = nav_pkg.robot_viz:main',
            'dwa_debug = nav_pkg.dwa_debug:main',
            'lidar_scan_check = nav_pkg.lidar_scan_check:main',
            'nav_node_simple = nav_pkg.nav_node_simple:main',
            'nav_node_puzzlebot = nav_pkg.nav_node_pb:main',
            'nav_node_bug2 = nav_pkg.nav_node_bug2:main',
        ],
    },
)
