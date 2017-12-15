# bevel

A minimal tool for ordinal logistic regression.

```python
from bevel import OrdinalRegression

orf = OrdinalRegression()
orf.fit(X, y)
orf.print_summary()
```