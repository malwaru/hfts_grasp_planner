from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from Cython.Build import cythonize

# fetch values from package.xml
setup_args = generate_distutils_setup(
    # ext_modules=cythonize("python-src/hfts_grasp_planner/placement/reachability/cpose_distance.pyx"),
    packages=['hfts_grasp_planner'],
    package_dir={'': 'python-src'},
    requires=['rospy', 'numpy', 'yaml', 'rtree', 'tf', 'stl',
              'sklearn', 'scipy', 'igraph', 'openravepy', 'rospkg']
)

setup(**setup_args)
