# QEI-Net: A Deep learning-based automatic quality evaluation index for ASL CBF Maps
![Python Version](https://img.shields.io/badge/python-3.10.12-blue.svg)
![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.16.1-brightgreen.svg)
![Ubuntu Version](https://img.shields.io/badge/ubuntu-22.04.04-orange.svg)
![NVIDIA GPU](https://img.shields.io/badge/NVIDIA-Tesla_T4-76B900.svg)
![CUDA Version](https://img.shields.io/badge/CUDA-12.4-blue.svg)


Xavier Beltran Urbano,  Sudipto Dolui and John A Detre

## Setting Up the Project
1. Clone the project:
  ```bash
  git https://github.com/xavibeltranurbano/QEI-Net/tree/main
  cd DL-Ensemble-Brain-Tissue-Segmentation
  ```
2. Using a virtual environment is strongly recommended.
```bash
python -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```
## Reproducing the Results
The project utilizes the following folder structure
```bash
DL-Ensemble-Brain-Tissue-Segmentation/
├── data
│   └── Training_Set
│       ├── IBSR_01
│       │   ├── IBSR_01.nii.gz
│       │   └── IBSR_01_seg.nii.gz
│       ├── IBSR_03
│       └── ...
├── src
│   ├── configuration.py
│   ├── ...
│   ├── networks
│   │   ├── denseunet.py
│   │   └── ...
```
To reproduce the results, execute the following scripts:
```bash
python src/main.py
```
This will train and validate the model specified in "networkName" and store it in results/.

## Dataset

## Methodology

## Results

### · Quantitative Results

#### Single Model Results

#### Ensemble Results


### · Qualitative Results


## Conclusion
