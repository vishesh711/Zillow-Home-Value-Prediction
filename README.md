# Zillow Home Value Prediction

This project involves predicting the log error of Zillow's home value estimates. We employ multiple regression techniques including Linear Regression, XGBoost, and LightGBM to improve the prediction accuracy.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing and EDA](#data-preprocessing-and-eda)
- [Linear Regression](#linear-regression)
- [Gradient Boosting Models](#gradient-boosting-models)
- [Combining Predictions](#combining-predictions)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

## Introduction
This project aims to predict future log error in home value estimates using various machine learning techniques. The datasets used are Zillow's properties data and train data from 2017. We will explore Linear Regression and advanced ensemble methods like XGBoost and LightGBM.

## Data Preprocessing and EDA
The initial steps involve data loading, handling missing values, and exploratory data analysis (EDA) to understand the data better.

- **Data Loading**: 
  ```python
  project = pd.read_csv("properties_2017.csv", low_memory=False)
  train = pd.read_csv("train_2017.csv", parse_dates=["transactiondate"])
  ```

- **Handling Missing Values**: Drop columns with excessive missing values and fill remaining with appropriate statistics.

- **EDA**: Use correlation matrices and visualizations to understand relationships between features.

## Linear Regression
Linear Regression is used to find a linear relationship between dependent and independent variables. The model showed a low variance score, indicating it does not fit well with our dataset.

## Gradient Boosting Models
We employed XGBoost and LightGBM, which are more robust and handle complex relationships better than Linear Regression.

### XGBoost
XGBoost is known for its performance and speed. We trained two models with different parameters and combined their predictions.

### LightGBM
LightGBM grows trees leaf-wise, which often leads to better accuracy. We processed the data accordingly and trained the model.

## Combining Predictions
To leverage the strengths of different models, we combined predictions from XGBoost, LightGBM, and a baseline model using weighted averages. This ensemble approach typically yields better results.

```python
# Combining predictions
XGB_WEIGHT = 0.6415
BASELINE_WEIGHT = 0.0056
OLS_WEIGHT = 0.0828
XGB1_WEIGHT = 0.8083

xgb_pred = XGB1_WEIGHT * xgb_pred1 + (1 - XGB1_WEIGHT) * xgb_pred2
lgb_weight = (1 - XGB_WEIGHT - BASELINE_WEIGHT) / (1 - OLS_WEIGHT)
xgb_weight0 = XGB_WEIGHT / (1 - OLS_WEIGHT)
baseline_weight0 = BASELINE_WEIGHT / (1 - OLS_WEIGHT)
pred0 = xgb_weight0 * xgb_pred + baseline_weight0 * BASELINE_PRED + lgb_weight * p_test
```

## Results
The final predictions were combined and saved to a CSV file for submission.

```python
submission.to_csv('sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
```

## Conclusion
Linear Regression was not suitable for this dataset, as indicated by a low variance score. Gradient Boosting models like XGBoost and LightGBM provided better accuracy. Combining predictions from multiple models yielded improved results.

## Dependencies
- numpy
- pandas
- seaborn
- matplotlib
- xgboost
- lightgbm
- sklearn

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/zillow-home-value-prediction.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or script:
   ```bash
   jupyter notebook zillow_home_value_prediction.ipynb
   ```

## Contributors
- Vishesh Kumar

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to this project by submitting a pull request or reporting any issues. Happy coding!
