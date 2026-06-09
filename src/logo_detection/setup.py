from setuptools import find_packages, setup

package_name = 'logo_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # modelos y json viajan junto al modulo (los nodos los cargan con Path(__file__).parent)
    package_data={package_name: ['*.pt', '*.json']},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='your_email@tec.mx',
    description='YOLO logo detection and dock alignment nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'logo_detection = logo_detection.logo_detection:main',
            'logo_allignment = logo_detection.logo_allignment:main',
            'logo_callibration = logo_detection.logo_callibration:main',
            'logo_visual_callibration = logo_detection.logo_visual_callibration:main',
        ],
    },
)
