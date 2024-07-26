# In2Research-2024

This repository contains the code written for the In2Research placement of
@yousuf-adialwa20 in the Neuroinformatics Unit during the summer of 2024.

## Summary
The overall goal of this project was to to use
[`movement`](https://movement.neuroinformatics.dev/) for the pre-processing
and exploratry analysis of data acquired by the placement co-supervisor 
[Shanice Bailey](https://www.sainsburywellcome.org/web/people/shanice-bailey).

The dataset in question was acquired to investigate the social interaction
between pairs of interacting mice and it consists of short videos.
The videos had been processed with the pose estimation software
[DeepLabCut](https://www.mackenziemathislab.org/deeplabcut) to extract the
x, y positions of various body parts at each video frame.

The extracted bodypart positions across time - referred to as predicted
pose tracks - were cleaned using `movement` and then various metrics were
calculated to quantify aspects of the social interaction between the mice.

## Setup

To run the code in this repository, including the jupyter notebooks, you will
need to create a new [conda](https://docs.anaconda.com/miniconda/) environment
with the required dependencies, which are listed in the 
[environment.yml](./environment.yml) file.

To create the required environment, follow these steps:

1. Clone the repository and navigate to the repository folder:
    ```bash
    git clone https://github.com/neuroinformatics-unit/In2Research-2024
    cd In2Research-2024
    ```
2. Create the conda environment from the environment file and activate it:
    ```bash
    conda env create -n In2Research2024 -f environment.yml
    conda activate In2Research2024
    ```

Note that we have named the environment `In2Research2024`, but you can choose
any name you like. 

Beware that the environment specifically contains
`movement==0.0.19`, because that's the version of the `movement` package that
was used during the placement. If you are using newer versions of the package,
some of the code may need adjustments, as the syntax is likely to have changed.