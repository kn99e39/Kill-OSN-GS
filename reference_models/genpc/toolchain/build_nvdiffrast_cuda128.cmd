@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_PATH_V12_8=%CUDA_HOME%"
set "DISTUTILS_USE_SDK=1"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%PATH%"

where nvcc
nvcc --version
where cl
cl
set > C:\Projects\Kill-OSN-GS\reference_models\genpc\toolchain\cuda128_msvc_environment.txt

cd /d C:\Projects\Kill-OSN-GS\reference_models\genpc\dependencies\nvdiffrast
C:\Projects\Kill-OSN-GS\reference_models\genpc\env-blackwell\Scripts\python.exe -m pip install --no-build-isolation --no-deps .
exit /b %errorlevel%
