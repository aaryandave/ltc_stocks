# Stock Market Trend Prediction with LSTM Neural Networks

This project uses Liquid Neural Networks to predict the closing price of a stock. The goal is to evaluate the performance of LNNs for stock market prediction. Currently, the model is trained on historical data for Google stock (GOOGL).

## Overview

### Files

- `GOOGL.csv`: Contains historical data with Date, Open, High, Low, Close, Adj Close, and Volume for Google stock.
- `stock_model_(16, 1).h5`: Saved LTC model for predicting stock trends.
- `train_ltcs.py`: Python script for training the LTC model.

### Requirements

- Python 3
- TensorFlow
- pandas
- ncps (LTC library)
- matplotlib
- seaborn

### Usage

1. Install necessary dependencies:

    ```
    pip install tensorflow pandas ncps matplotlib seaborn
    ```

2. Clone the repository and navigate to the project folder.

3. Place your historical stock data in `GOOGL.csv`.

4. Run the `train_ltc.py` Python script:

    ```
    python stock_prediction.py
    ```

### Notes

- The project focuses on LTC Neural Networks to forecast stock market trends based on historical data.
- Experiment with model configurations and hyperparameters for better predictions.
