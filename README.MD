# Pediatric Appendicitis Diagnostic Model

This repository contains an end-to-end machine learning project. The goal of this project is to develop a model that predicts appendicitis in pediatric patients using structured clinical data.

---

## Project Goal

Appendicitis is one of the most common surgical emergencies in children. The diagnostic challenge is balancing two risks:

- Missing a real appendicitis case (which can lead to perforation and complications)
- Operating unnecessarily on a child without appendicitis

For this reason, the primary focus of this project was not just overall accuracy, but **recall for appendicitis** — minimizing missed cases.

---

## Dataset

The project uses the **Regensburg Pediatric Appendicitis dataset**, collected at Children’s Hospital St. Hedwig in Regensburg, Germany (2016–2021).

- 782 pediatric patients  
- ~58 clinical, laboratory, and ultrasound features  
- Includes symptom scores, blood tests, and imaging findings  

This dataset is publicly available and used here for educational and research demonstration purposes.

Public source:  
https://www.kaggle.com/datasets/joebeachcapital/regensburg-pediatric-appendicitis  

---

## What This Project Includes

This repository contains:

- `main.py` — full pipeline-based training and evaluation script  
- `data/app_data.xlsx` — dataset file  
- Generated charts (distribution plots, model comparison, confusion matrix)

The project is implemented as a single executable Python script for clarity and reproducibility.

---

## Approach

### Data Cleaning
- Dropped columns with more than 50% missing values  
- Removed post-diagnosis variables (e.g., Length_of_Stay, Management) to avoid data leakage  
- Standardized diagnosis labels into binary format:
  - `0 = appendicitis`
  - `1 = no appendicitis`

### Preprocessing
All preprocessing was built inside a **scikit-learn Pipeline** to prevent leakage.

- Numerical features → median imputation  
- Categorical features → most frequent imputation + one-hot encoding  
- 80/20 stratified train-test split  

### Models Compared
- Decision Tree  
- Random Forest (100 estimators)  
- XGBoost  

Hyperparameter tuning was performed using 5-fold cross-validation with RandomizedSearchCV.

---

## Results

| Model | Accuracy |
|--------|----------|
| Decision Tree | 84.6% |
| Random Forest | 91.0% |
| XGBoost | 92.9% |
| Tuned XGBoost | 92.9% |

On the held-out test set (156 patients):

- 85 of 93 true appendicitis cases were correctly identified  
- Appendicitis recall ≈ 91%  

The most predictive feature was **Appendix Diameter**, which aligns with how clinicians evaluate ultrasound findings in practice.

---

## What I Learned

- Data leakage is easy to introduce and hard to notice. Using Pipelines prevents subtle errors.
- Accuracy alone is misleading in medical classification problems.
- More complex models do not automatically outperform simpler ones.
- Seeing a model independently identify clinically meaningful features was especially rewarding.

---

## Limitations

This dataset is publicly available and used here for educational and research demonstration purposes.  

The dataset comes from a single hospital and may not generalize to other populations.

---

## How to Run

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl

Then run:

python main.py

## Author

Lalitha Srihitha Prayaga  
M.S. Health Informatics  
Indiana University Indianapolis