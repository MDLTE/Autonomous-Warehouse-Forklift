from setuptools import find_packages, setup

package_name = 'voice_commands'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # codebook y modelos hmm viajan junto al modulo
    package_data={package_name: ['*.npy', '*.pkl']},
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
    description='HMM voice command recognition node for PuzzleBot',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'voice_puzzlebot = voice_commands.voice_puzzlebot:main',
        ],
    },
)
