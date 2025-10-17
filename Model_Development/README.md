# Home Credit Default Risk — Modeling Pipeline

This repository covers model development stage — from **base model optimization** to **multi-strategy ensembling** — ensuring both reproducibility and interpretability.

---

## Project Structure

### `model.ipynb`
 The main development notebook, documenting the full modeling process (recommended starting point).

---

### `Base_Models_Para_Optimize/`
Source code of all **base models** with integrated grid search or small-scale hyperparameter optimization.

---

### `Base_Models_Weighted_Comparison/`
Experimental comparison of **five linear-type base models**, evaluating the impact of **sample weighting vs. no weighting** on performance.

---

### `Results_Pictures/`
Our main results in the stage of model development.

---

## Other Files

| File | Description |
|------|--------------|
| **`Final_submission.csv`** | The final blended submission file — best overall model ensemble. |
| **`README.md`** | This documentation file. |

---

## Suggested Reading Order

To follow the logical modeling workflow:

1. `model.ipynb` — overview and complete pipeline  
2. `Base_Models_Para_Optimize/` — base model training
3. `Base_Models_Weighted_Comparison/`— weighting effect experiments  
4. `Final_submission.csv` — final output  

---

## Environment

- Python ≥ 3.9  
- LightGBM, XGBoost, CatBoost, Scikit-learn, Optuna, Pandas, NumPy, Matplotlib, Seaborn  
- Compatible with macOS (Apple Silicon), Linux, and AutoDL environments  

---

## Contact

If you need our data in this part or have any other questions, please contact the course teaching team.