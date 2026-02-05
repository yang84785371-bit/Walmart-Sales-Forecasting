# Walmart-Sales-Forecasting
This project implements an end-to-end sales forecasting pipeline based on the Walmart weekly sales dataset.   The goal is to systematically compare traditional machine learning models and deep learning sequence models under a unified, reproducible framework.

## Project Overview
The pipeline covers data cleaning, feature engineering, dataset splitting, model training, evaluation, and result visualization.  
Multiple modeling paradigms are explored, including tree-based models and sequence models, to analyze their suitability for short-horizon retail demand forecasting.

## Data
- Dataset: Walmart Weekly Sales
- Granularity: Store × Department × Week
- Time span: 2010-02 to 2012-10

## Methods
### Feature Engineering
- Lag features (1, 2, 4, 8, 12 weeks)
- Rolling statistics (mean / std)
- Calendar features (week, month, holiday)
- Strict time-based train/validation split

### Models
- Random Forest
- XGBoost (log1p target + early stopping)
- LSTM
- Transformer
- Enhanced DL variants with Store / Department embeddings and residual calibration

### Evaluation Metrics
- MAE
- RMSE
- MAPE
- WAPE

## Results
Tree-based models (Random Forest and XGBoost) achieved the best overall performance, while LSTM and Transformer models showed limited gains under short-window, high-heterogeneity conditions.  
Embedding-based deep learning variants improved stability but did not surpass tree-based baselines, highlighting the importance of explicit feature engineering in this task.

## Visualization
- Total sales prediction vs. ground truth over time
- Predicted vs. true value scatter plots
- Absolute error distribution comparison across models

## Project Structure
```
walmart_sale_forecast/
├── data/ # cleaned data, features, train/valid splits
├── scripts/ # model training scripts
├── tools/ # data processing, evaluation, visualization
├── output/ # metrics, predictions, figures
```
## Key Takeaways
- Explicit lag and rolling features are highly effective for short-horizon retail forecasting.
- Tree-based models are strong baselines for heterogeneous tabular time-series data.
- Deep learning models require explicit entity-level modeling (Store / Dept) to narrow the performance gap.

## Output
<img width="960" height="720" alt="valid_abs_error_hist" src="https://github.com/user-attachments/assets/63ae1e60-4fa0-4fe1-88ec-c54b0b177e3b" />
<img width="960" height="720" alt="valid_scatter_true_vs_pred" src="https://github.com/user-attachments/assets/d1dcbabc-280d-46f5-87e1-d0b034a3eb9a" />
<img width="960" height="720" alt="valid_total_sales_timeseries" src="https://github.com/user-attachments/assets/5f4c4184-4944-4b97-86dc-5103253daca9" />




