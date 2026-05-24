# Large Language Models (LLM) enabled Physical Activity Classification (PAC) using Wearable Devices

Third Year Project for the University of East Anglia.
## Overview
This project aims to investigate the feasibility of using lightweight Large Language Models for
physical activity classification using wearable sensor data by comparing performance to a set
of benchmark traditional machine learning classifiers. Rather than change the internal structure
of the LLM, prompt engineering will be used to embed raw data into the model’s input. Three
different types of prompt engineering strategies will be used: zero-shot, few-shot and Chainof-Thought. As well as reviewing the overall ability of a lightweight Large Language Model
for classification, the effect of using differing prompting strategies and how it affects
classification accuracy will be looked at

## Models used
### Traditional Machine Learning Models
- k-Nearest Neighbours
- XGBoost
- Random Forest
- Multi-layer Perceptron

### Large Language Model
- Gemma-2B-it

### Prompt Engineering Strategies
- Zero-shot Prompting
- Few-shot Prompting
- Chain-of-Thought (CoT) Reasoning
- Random Forest + CoT Hybrid System

## Feature engineering
Sensor data taken from the Capture24 dataset was segmented using a sliding window with 50% overlap. The following features were extracted:
- X,Y,Z mean
- X,Y,Z std
- X,Y,Z min
- X,Y,Z max
- Magnitude mean
- Magnitude std
- RMS
- Jerk
- Dominant Frequency
- Label (mode)

## How to use the project
1. Download Capture24 dataset locally, extract CSV files and place in a local directory
   2. Change relevant directory paths in feature_preprocessing.py to Capture24 location
3. Run feature_preprocessing.py to create datasets for models
4. Run each machine learning model to fine-tune and assess models
   5. k_nearest_neighbours.py
   5. random_forest.py
   5. multilayer_perceptron.py
   5. xg_boost.py
6. run llm.py to assess prompt engineering
7. analyse graphs in graphs folder for results



## Packages
The project was developed using Python and requires the following libraries:
- Pandas
- Numpy
- SciKit
- Matplotlib
- Seaborn
- Pickle
- XGBoost
- Torch
- Transformers

Additional requirements:
- Python 3.10+
- CUDA compatible GPU for LLM inference
- HuggingFace authorisation for Gemma-2b-it

## Results Summary
Overall, the findings of this project present that lightweight Large Language Models are not
capable of physical activity classification through standalone prompting at this level. However,
they still could offer valuable insights if fine-tuned to a relevant domain in a hybrid system like
the Random Forest + Chain-of-Thought system, where a Large Language Model may be able
to offer valuable explanations, arguments or interpretations of data to aid someone in
completing a task or classifying data

## Author
Developed by Liam Barry, 
University of East Anglia
