# Nested-NP-Identification
Nested noun phrase identification in English and Swedish sentences using BERT.

# Data Files
  - The OntoNotes data corpus and Swedish Talbanken data corpus are needed to train the model in this code and run experiments.
    
#Steps to run the code
 ## Getting the data ready
  1. Place the data corpus files in DataFiles folder in the main directory.
  2. Run the files ontonotes_dataloader.py and swedish_treebank_loader.py to create datasets for training the BERt and Swedish BERT models respectively.
## Fine-tuning the models
  1. Create Models folder in the mian directory to store the fine tuned models.
  2. Run the scripts 'finetuning_BERT.py' and 'finetuning_SWE_BERT.py' in the Code folder to fine-tune and store the English and Swedish models respectively.
## Evaluating the models
  1. Create Results folder in the main directory to store plots and results.
  2. Run the files 'evaluate_BERT.py' and 'evaluate_SWE_BERT.py' in the Code folder to obtain evaluation results and relevant plots. Follow comments in the code to appropriately name the plots.

