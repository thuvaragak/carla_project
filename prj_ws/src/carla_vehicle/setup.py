from setuptools import find_packages, setup

package_name = 'carla_vehicle'

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
    maintainer='thuvaraga',
    maintainer_email='thuvaraga@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vehicle = carla_vehicle.vehicle:main',
            'objdetection = carla_vehicle.obj_detection:main',
            'adas_vehicle_detction = carla_vehicle.adas_vehicle_detction:main'

        ],
    },
)
