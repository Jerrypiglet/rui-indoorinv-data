# Configuring the environment

The code is tested with Pytorch=1.13 on a M1 Mac and a Ubuntu machine with CUDA. Not tested with other versions or platforms.

``` bash
conda create --name py310 python=3.10 pip
conda activate py310
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Note on Pytorch
For Arm64 Mac, you will need to install Pytorch with GPU-acceleration with MPS. For Pytorch 2.x, MPS should be ready with pip installation (according to https://pytorch.org/get-started/locally/). For Pytorch 1.1x, 

``` bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```
## Note on Blender/bpy

For `bpy` (Blender python API), which is required to render synthetic scenes, `pip install bpy` should be sufficient on Linux. For Mac, if pip install does not work, try this: 

``` bash
brew install libomp
mkdir ~/blender-git
cd ~/blender-git
git clone https://git.blender.org/blender.git
cd blender
make update
mkdir ../build
cd ../build

# ../blender/CMakeLists.txt
include_directories("/usr/local/include" "/opt/homebrew/opt/libomp/include")
link_directories("/usr/local/lib" "/opt/homebrew/opt/libomp/lib")

ccmake ../blender
  WITH_PYTHON_INSTALL=OFF
  WITH_AUDASPACE=OFF
  WITH_PYTHON_MODULE=ON

make -j10
python ../blender/build_files/utils/make_bpy_wheel.py ./bin/
pip install bin/bpy-***.whl # --force-reinstall if installed before
```

<!-- Install OpenEXR on mac:

``` bash
brew install openexr
brew install IlmBase
export CFLAGS="-I/Users/jerrypiglet/miniconda3/envs/py310/lib/python3.10/site-packages/mitsuba/include/OpenEXR"
# export LDFLAGS="-L/opt/homebrew/lib"
pip install OpenEXR
``` -->

Hopefully that was everything. 
## Note on Mitsuba 3 on Arm64 Mac
On Mac, make sure you are using am Arm64 Python binary, installed with Arm64 version of conda for example. Check your Python binary type via:

``` bash
file /Users/jerrypiglet/miniconda3/envs/py310/bin/python
# yields: /Users/jerrypiglet/miniconda3/envs/or-py310/bin/python: Mach-O 64-bit executable arm64
```

Then install llvm via:

``` bash
brew install llvm
```