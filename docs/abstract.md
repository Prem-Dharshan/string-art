20XW97 - Package Abstract
Computer Vision approach to String Art

22PW01 - AJAY H
22PW29 - PREM DHARSHAN D

Abstract
String art builds an image from a single thread stretched between pins arranged on a circular/rectangular frame. In this project, we develop a simple computer vision pipeline that converts photographs into thread paths. Input images are preprocessed through grayscale conversion, contrast enhancement, and Gaussian smoothing, after which a greedy algorithm selects lines one at a time to best match the target image. We use facial landmark detection to build importance maps that preserve detail in key regions such as the eyes and mouth. The method is also extended to color using k-means palette selection and Floyd–Steinberg dithering.
We evaluate the generated outputs using PSNR and SSIM, both on the direct render and after blurring to simulate viewing distance. We further study how the number of lines and the use of importance weighting affect output quality, and we visualize the image forming thread by thread. The pipeline produces a pin sequence that can be used for physical fabrication.
