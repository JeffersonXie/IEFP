# IEFP
This repo is the official implementation of our paper entitled "Implicit and Explicit Feature Purification for Age-invariant Facial Representation Learning"

## Requirements
- Pytorch >= 1.1
- Python >= 3.7
- CUDA enabled computing device

## Usage
### Download face datasts
1. Please download the following four processed face datasets for evaluation purposes. They include: [morph2](https://drive.google.com/file/d/1P6bEBCax0P7GcfwGlZu4aFpioX16cZI9/view?usp=sharing), [FGNET](https://drive.google.com/file/d/1A2MHRUkNlIk5l7grDRY84vZ8egT4UGXN/view?usp=sharing), [CACDVS](https://drive.google.com/file/d/1XldmKL8s8_nR5Owxdcrq5ZGM0PJiP-_s/view?usp=sharing) and [CALFW](https://drive.google.com/file/d/1nuW8g2irSJFFrOG32_J3xRaOXZxHe62R/view?usp=sharing). All of them should be unziped and then put in the folder named "face_dataset". In addition, regarding the downed CACDVS and CALFW, they need to be combined with correponding txt file to form a new subfolder which is named accordingly.
2. You also need to do
