
# Face Swap Pipeline

This repository contains a face swapping pipeline built on 3DDFA_V2 Morphable Face Models (https://github.com/cleardusk/3DDFA_V2). The pipeline processes an input face image provided by the user, generates a corresponding 3D model, tracks face models within a video, and swaps the source face onto one or more characters in the video.

DISCLAIMER: This project is unfinished. It is likely that the final product will look very different from how it does now.

## License
All original work is reserved. All previously copyrighted work is used under the following MIT license:

Copyright (c) 2017 Max deGroot, Ellis Brown
Copyright (c) 2019 Zisian Wong, Shifeng Zhang
Copyright (c) 2020 Jianzhu Guo, in Center for Biometrics and Security Research (CBSR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Overview

This pipeline utilizes 3DDFA_V2 Morphable Face Models to perform face swapping. It generates a 3D model from a provided face image, tracks face models within a video, and replaces the faces in the video with the input face. All files are designed to be run in a linux terminal, with the exception of view_obj.py which has been designed for a powershell terminal.

## Demonstration

### Input Face

<img src="examples/inputs/cillian.jpg" alt="Description of the image" width="300" />

### Source Video

![Source Video](examples/inputs/severance.gif) 

### Face Swap Result

![Swap Result](examples/results/severance_cillian_swapped_corrected.gif) 

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. **Activate the virtual environment**  
   The requirements for this project have been pre-installed into a virtual environment running Python 3.9. To use the code, activate the venv by running the following command in your terminal:
   ```bash
   source venv39/bin/activate
   ```
4. **Run the Pipeline**  
   Use image_swap or video_swap, along with your own images to create a basic swap.

For any questions or contributions, please refer to the repository guidelines or open an issue.
```
