# AM 221 Project
This repository contains code, output, and configuration files for my project, `Towards Inducing Abstention in Certified Defense (Maybe)`.

### Structure
The steps to run the project are given in [run_project.sh](run_project.sh); the structure is as follows:

    - Query black-box model to generate training data for substitute model
    - Train substitute model
    - Generate adversarial images by attacking substitute model
    - Attack black-box model with adversarial images

### Code
Much of the code for this project is based heavily on work by others. The black-box model code was taken from a [repository](https://github.com/locuslab/smoothing) provided by the authors of the original randomized smoothing paper. The code underlying the substitute model generation was adapted from [this](https://github.com/icpm/pytorch-cifar10). I wrote the code for the adversarial attacks and figure generation.

### Configuration
Every step of the project, except for figure generation, was run on Google Cloud with the following specifications:

    - Boot Disk - Deep Learning Image: PyTorch 1.0.0 and fastai m23 CUDA 10.0 with 100 GB
    - CPU/RAM - 8 vCPUs and 52 GB RAM
    - GPU - 1 x NVIDIA Tesla P100

I do not have a software configuration file for this, but it should not be too difficult to install the necessary software by looking at the `import` statements at the top of the code that is run.

Figure generation for the paper takes place in [paper_analysis/create_figures.py](paper_analysis/create_figures.py). The [project_env.yaml](project_env.yaml) file is a conda environment configuration file that will provide all the necessary software for the figure generation step.
