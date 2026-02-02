# Overfitting Prevention in Trading ML

## The Problem

In trading, overfitting is deadly:
- Strategy looks amazing in backtest
- Fails completely in live trading
- You've fit noise, not signal

## Key Techniques

### 1. Walk-Forward Optimization

Never optimize on data you'll test on:

```python
def walk_forward_cv(data, train_size=252, test_size=63):
    """
    Rolling train/test split for time series.
    """
    splits = []
    for i in range(0, len(data) - train_size - test_size, test_size):
        train = data[i:i+train_size]
        test = data[i+train_size:i+train_size+test_size]
        splits.append((train, test))
    return splits
```

### 2. Cross-Validation for Time Series

Use purged and embargoed CV:

```python
def purged_kfold(data, n_splits=5, purge_gap=5):
    """
    Time series CV with gap between train/test.
    """
    fold_size = len(data) // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size
        
        # Purge: remove data close to test set
        train = pd.concat([
            data[:test_start - purge_gap],
            data[test_end + purge_gap:]
        ])
        test = data[test_start:test_end]
        yield train, test
```

### 3. Regularization

Penalize model complexity:

```python
from sklearn.linear_model import Ridge, Lasso

# L2 regularization
model = Ridge(alpha=1.0)

# L1 regularization (feature selection)
model = Lasso(alpha=0.1)
```

### 4. Feature Selection

Less is more. Use:
- Information ratio ranking
- Recursive feature elimination
- Domain knowledge filtering

```python
def select_features(X, y, max_features=10):
    """
    Select top features by information ratio.
    """
    from sklearn.feature_selection import mutual_info_regression
    
    mi_scores = mutual_info_regression(X, y)
    top_idx = np.argsort(mi_scores)[-max_features:]
    return X.iloc[:, top_idx]
```

### 5. Ensemble Diversity

Combine different model types (AQTIS approach):
- Random Forest
- Linear (Ridge)
- LSTM
- Rules-based

Overfit models tend to fail in different ways, so averaging helps.

## Red Flags for Overfitting

| Warning Sign | What It Means |
|--------------|---------------|
| Sharpe > 3 in backtest | Too good to be true |
| Train >> Test performance | Classic overfit |
| Many parameters | More degrees of freedom to fit noise |
| Data snooping | Tested many strategies, only showing winners |
| No transaction costs | Unrealistic P&L |

## AQTIS Safeguards

1. **Minimum sample size**: 100+ trades before trusting statistics
2. **Out-of-sample validation**: Always hold out recent data
3. **Degradation monitoring**: Track live vs backtest performance
4. **Model simplicity**: Prefer interpretable models
5. **Ensemble weights**: Adapt based on live performance

## References

- de Prado, M.L. "Advances in Financial Machine Learning"
- Bailey, Borwein, et al. "Pseudo-Mathematics and Financial Charlatanism"
