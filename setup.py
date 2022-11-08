from setuptools import setup

package_name = 'hagen_control_2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kahlflekzy',
    maintainer_email='felixsigit@gmail.com',
    description='A package for my tutorials on controlling the hagen wheeled robot.',
    license='BSD-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = hagen_control_2.controller:main',
            'controller2 = hagen_control_2.controllers:main',
            'path_follower = hagen_control_2._controller:main',
            'ekf = hagen_control_2.ekf_control:main'
        ],
    },
)
