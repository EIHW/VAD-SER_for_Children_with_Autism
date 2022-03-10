# Voice Activity Detection-based Speech Emotion Recognition for children with autism

(c) 2017-2020 Manuel Milling, Alice Baird: Universität Augsburg Published under GPLv3, see the LICENSE.md file for details.

Please direct any questions or requests to Manuel Milling (milling.manuel@gmail.com). 

## Citing
If you use the in this repository in your research work, you are kindly asked to acknowledge the use in your publication.

> Milling, Manuel, Alice Baird, Katrin D. Bartl-Pokorny, Shuo Liu, Alyssa M. Alcorn, Jie Shen, Teresa Tavassoli et al. "Evaluating the Impact of Voice Activity Detection on Speech Emotion Recognition for Autistic Children." Frontiers in Computer Science 4 (2022): 837269.

Bibtex
```
@ARTICLE{milling2022_vad-ser,
AUTHOR={Milling, Manuel and Baird, Alice and Bartl-Pokorny, Katrin D. and Liu, Shuo and Alcorn, Alyssa M. and Shen, Jie and Tavassoli, Teresa and Ainger, Eloise and Pellicano, Elizabeth and Pantic, Maja and Cummins, Nicholas and Schuller, Björn W.},   
TITLE={Evaluating the Impact of Voice Activity Detection on Speech Emotion Recognition for Autistic Children},      
JOURNAL={Frontiers in Computer Science},      
VOLUME={4},     
YEAR={2022},      
URL={https://www.frontiersin.org/article/10.3389/fcomp.2022.837269},       
DOI={10.3389/fcomp.2022.837269},      
ISSN={2624-9898},   
}
```
## Overview

This repository offers two components, explained in more detail in the paper referenced above: the first component is a  voice activity 
detection component, which can be trained in particular for vocalisations of children with autism, for general 
vocalisations or similar tasks. The second component is a continuous speech emotion recognition component, which is 
supposed to be used after preprocessing of the data with the VAD systems.
Due to privacy regulations the data used in the referenced paper cannot be published.

## Usage
The implementation is based on python 3.8.10 tensorflow 2.3.1. Additional requirements can be installed with `requirements.txt`

### Voice Activity Detection
The RNN-LSTM model used for the VAD task can be loaded from `src.models.audio_feature_models.py`.
The model expects input features (10ms LLDs recommended), which can be normalised and extracted for instance with the opensmile package https://github.com/audeering/opensmile
and `extract_features_command_line.py`. The script `main_VAD` contains code for creating the dataset dependent datagenerator
and the datageneration as well as the training, evaluation and inference of the models.

### Speech emotion recognition
The SER task is implemented in a similar as the previous task, however it uses the `main_ser.py` for datageneration, 
training and evaluation of the models. The features (1s eGeMAPs functionals) and sequence-based data generation and according 
CCC evaluation vary in comparison to the VAD task. 
