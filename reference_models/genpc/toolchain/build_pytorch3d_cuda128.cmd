@echo off
setlocal

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64
if errorlevel 1 exit /b %errorlevel%

set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "CUDA_PATH_V12_8=%CUDA_HOME%"
rem CUDA 12.8 ships CCCL 2.7.0 (Thrust/CUB 2.0.7). Use the official
rem matched CUB tree instead of the standalone CUB 2.1.0 checkout, whose
rem legacy device header includes a Thrust private header removed in 2.0.7.
set "CUB_HOME=C:\Projects\Kill-OSN-GS\reference_models\genpc\dependencies\cccl_2_7_0"
set "DISTUTILS_USE_SDK=1"
set "FORCE_CUDA=1"
set "MAX_JOBS=2"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\lib\x64;%PATH%"

where nvcc
nvcc --version
where cl

cd /d C:\Projects\Kill-OSN-GS\reference_models\genpc\dependencies\pytorch3d
C:\Projects\Kill-OSN-GS\reference_models\genpc\env-blackwell\Scripts\python.exe -m pip install --no-build-isolation --no-deps .
exit /b %errorlevel%
