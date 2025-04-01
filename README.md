
# Face Swap Pipeline

This repository contains a face swapping pipeline built on 3DDFA_V2 Morphable Face Models. The pipeline processes an input face image provided by the user, generates a corresponding 3D model, tracks face models within a video, and swaps the source face onto one or more characters in the video.

## License


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

This pipeline utilizes 3DDFA_V2 Morphable Face Models to perform face swapping. It generates a 3D model from a provided face image, tracks face models within a video, and replaces the faces in the video with the input face.

## Demonstration

### Input Face

![Input Face](examples/inputs/cillian.jpg)  
*Replace `path/to/face_image.jpg` with the correct path to the input face image.*

### Source Video

<video width="640" height="360" controls>
  <source src="examples/inputs/severance.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>  
*Replace `path/to/source_video.mp4` with the correct path to the source video.*

### Face Swap Result

<video width="640" height="360" controls>
  <source src="examples/results/severance_cillian_swapped_corrected.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>  
*Replace `path/to/swapped_video.mp4` with the correct path to the face swap result video.*

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. **Install Dependencies**  
   Refer to the instructions in [INSTALL.md](./INSTALL.md) for dependency installation.
3. **Run the Pipeline**  
   Follow the guidelines provided in [USAGE.md](./USAGE.md) to execute the pipeline on your input data.

For any questions or contributions, please refer to the repository guidelines or open an issue.
```
