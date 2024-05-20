# NER_training

This repository can be used to train NER models.

## Setup Instructions

1. **Install Dependencies**:
   Ensure you have all the necessary dependencies installed by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Training Data**:
   The training data must be kept in the `data` folder combined into a single file named `train.csv`. The CSV file must follow the same structure as [this HF dataset](https://huggingface.co/datasets/ksgr5566/ner).

3. **Prepare Test Data (Optional)**:
   Another CSV file named `test.csv` can be added to the `data` folder for evaluation metrics. If `test.csv` is not present, the evaluation will be done on `train.csv` itself. The evaluation results will be populated in the `logs` folder.

4. **Run Training**:
   Once the data is prepared, run the training script using the following command:
   ```bash
   python3 '/trainer/train_bert.py'
   ```

Follow these instructions to set up and train your NER model using this repository.
