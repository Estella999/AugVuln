<<<<<<< HEAD
# AugVuln

## Overview

This replication package contains the dataset and code for our paper Root Cause Detection by Capturing Changed
code with Relational Graph Neural Networks.

This work struggle to solve the presence of labeling errors and characterizing noise in the code lines,
and imbalanced knowledge distribution within the code.AugVuln that integrates code quality detection with Relational Graph Neural Networks (RGNNs) to improve root cause detection by more effectively capturing code changes. Specifically, the AugVuln incorporates three key components: (1) the Vulnerability Confidence Extractor for detecting and cleaning erroneous labels; (2) the Code Generation Enhancer, which generates balanced synthetic test cases via Conditional Variational Autoencoders (CVAE); and (3) the Path-Aware Vulnerability Detector, which employs Relational Graph Neural Networks (RGNNs) to capture changes in code lines and integrates residual networks to learn detailed representations of code execution paths.Through the synergistic operation of these three components, AugVuln effectively handles code lines with noisy labels and imbalanced knowledge distribution, leading to significant improvements in vulnerability detection


## Directory Structure

### `code`
Contains all source code related to data processing, parser tools, and utility scripts.
AugVuln.py: The main code of training .
CGE.py: The main code of CGE .
run.py: Training framework.
data_preprocess.py:Processing source code data, generating abstract syntax tree (AST) paths, and preparing the dataset for training, validation, and testing.

- **`data_process`**
    
    - ProcessedData.py:Managing and manipulating preprocessed datasets, including feature selection and data synthesis functionalities.
    - textdata.py:Implements data processing, feature extraction, dataset management, and evaluation functions.essential for preparing the dataset and assessing model performance.

- **`parserTool`** the tool of Tree-sitter.

- **`utils`**  parses a source code's abstract syntax tree to construct a control flow graph.

### `dataset` 
Stores datasets used for training and evaluating models.
Please download the dataset at the specified link in dataset.txt

- **`cdata`**
  - **Subdirectories:**
    - `D2A/`: Dataset from D2A.
    - `Diversuval/diversuval/`: Dataset from Diversuval.
    - `VDR/`: Dataset from Big-Vul、Devign、REVEAL.

- **`javadata`**
  - **Subdirectories:**
    - `CSGCD/`: Java dataset.

### `models`
Contains pretrained models and scripts for model training.

- **`codebert`** the pre-trained CodeBERT model
- Please download the dataset at the specified link in codebert.txt

- **`CVAE`** Conditional Variational Autoencoder (CVAE) model files.

## Replication Process
### Step 1: Clone the Repository

`cd AugVuln`

### `Step 2: Environment Preparation`
- Python: 3.8
- Pytorch: 1.11.0+cu113
- networkx: 2.8.5
- numpy: 1.22.3
- scikit-learn: 1.1.1
- scipy: 1.8.1
- tree-sitter: 0.20.1
- cleanlab:0.1.1
  
### `Step 3: Preprocess the Dataset`
`cd code`

`bash data.sh`

Please modify data.sh to set the appropriate dataset subdirectory (e.g., cdata/D2A/, cdata/VDR/, etc.).
Repeat this step if you want to generate .pkl files for multiple datasets.

###`Step 3:Training & Evaluation`
---
`bash train.sh`

`bash test.sh`

=======
# AugVuln
