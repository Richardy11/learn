# RESCU Research Kit

## Description

This repository contains IBT's research framework that allows for real time data streaming from a variety of input sources, data processing and data visualization modules, and virtual and physical prosthesis control options. Code in this repository has been specifically tested with Python 3.6.8 on Windows 10.
This repository relies on the existense of files not included in the repository, see installation guide to set up the framework.

## Installation Guide (ideal way)
The framework is designed to run on a virtual environment and requires no further components to install. However, there might be cases where the virtual environment does not behave well with pre-existing Python installations and libraries, in which case follow the bare-bones installation guide.

1. Install Python (3.6.8) for your operating system from [here](https://www.python.org/downloads/release/python-368/).

2. Download the following zip file and decompress contents into the repository root folder. Be advised that these files are included in the .gitignore and will not be source controlled by Bitbucket/GIT.
       [here](https://www.dropbox.com/s/g3xnumvlkac9h9m/Research_platform_offline_components.zip?dl=0)

3. For local streaming device addresses and names, edit the [ROOT]\code\local_address.py file which is included in the zip and the .gitignore.

## FAQ

1. VSCode is bad at detecting the virtual environment on its own. Online documentation suggests you go to File -> Preferences -> Settings -> Search 'python.pythonPath' and change the Workspace path to the location of the venv.

2. If this doesn't work, the next step is to go to the Explorer and look for the .vscode folder within the IBT folder. In this folder you can change the python.pythonPath to the same directory as above, save, and restart VSCode.

3. At this point VSCode should allow you to select venv36 as your interpreter and running the code should be fine.

## Installation Guide (bare-bones)

1. Install Python (3.6.8) for your operating system from [here](https://www.python.org/downloads/release/python-368/).

2. Launch the terminal (powershell for Windows, bash for Linux)

3. Upgrade the Python Package installer with the following terminal command:
        
        python -m pip install --upgrade pip

4. From the main directory, run the following code to install required Python packages:

        python -m pip install -r requirements.txt --user

## FAQ

Q1. Cannot install pybluez on Windows with the following error: 

        Could not find the Windows Platform SDK.
        Command "python setup.py egg_info" failed with error code 1...

A1. This error is because older versions of the pybluez installer depends on a specific version of the Windows Platform SDK and Visual Studio C++ compiler. This issue was supposed to be fixed in [pybluez-0.23](https://github.com/pybluez/pybluez/issues/180). If you still encounter this issue, please follow the following steps to install the previous version:

   1. Download [Visual Studio 2017](https://visualstudio.microsoft.com/vs/older-downloads/).
   2. Install VS2017 with **Desktop Development with C++ and Universal Windows Platform Development** selected. This should install Microsoft Visual C++ 14.0 compiler.
   3. Install the [Windows 10 SDK](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk/).
   4. Delete the cached version of the pybluez module using the following command:
   
            python -m pip cache remove pybluez
   
   5. Reinstall pybluez with the following command:
   
            python -m pip install pybluez==0.22 --user
   
   6. Both the Windows 10 SDK and VS2017 may be uninstalled via the Windows Control Panel.
            
