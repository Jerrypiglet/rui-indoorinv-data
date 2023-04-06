# https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html

conda create --name mitsuba-py38 python=3.8 pip
# remove /usr/local/include/pybind11/ or any other installations
conda install -c anaconda cmake
conda install -c conda-forge gcc
conda install -c conda-forge gxx
conda install sphinx
mkdir build
cd build
cmake -GNinja ..
ninja

# in your project path
source {path to mitsuba3/build}/setpath.sh
