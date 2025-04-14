# Kinematics library

I'd like a thing to do kinematics. I have one at work. I'd like to replicate some of it here.

The aim of this package is tracability. Where I just need to know that the high level functions are correct and behaving properly. As such matrices are built from base building blocks, and then impelemented separately for speed. The base implementation is tested against the speedy version.

In this way I can be sure that in as many cases as I can program, the speedy version is behaving the same as the base implementation (which I can check against lecture slides, wikipedia ect)

Next will follow some common reference frame operations such as switching from NED to ENU.
Or translating a bearing to an ENU frame + translating a relative bearing.

There will also be some standard ways to pull unit vectors from transformation matrices for easy plotting.

## Installation

1. Create python env: `python -m venv env`
2. Activate python env: `./env/Scripts/activate`
3. Install required packages: `pip install -r requirements.txt`
