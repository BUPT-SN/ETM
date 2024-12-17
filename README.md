### ETM
Code repository for the paper: "Leveraging the Dual Capabilities of LLM: LLM-Enhanced Text Mapping Model for Personality Detection" (AAAi 2025)

### Create Environment
First, create a Conda environment using the requirements.yml file:

```bash
conda env create -f requirements.yml
```

### Activate Environment  
After creation, activate the environment by replacing `<environment_name>` with your environment name.

### Dataset Preparation  
- **Kaggle Dataset**: Download the Kaggle dataset and place it in the `kaggle` folder.  
- **Pandora Dataset**: Download the Pandora dataset and place it in the `pandora/pandora_comments` folder.  
  - The `pandora` folder also contains dataset processing files.

### Pre-trained Models  
- **BERT-base-uncased**: Please download this model from Hugging Face or other platforms.  
- **Meta-Llama-3-8B-Instruct**: This model also needs to be downloaded independently.

## Run

Obtain Llama3 embeddings.

```bash
python llama3_features_tensor.py
```

Run the training script:

```bash
python train.py
```
