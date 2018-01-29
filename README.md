# bevel

Ordinal regression refers to a number of techniques that are designed to classify inputs into ordered (or ordinal) categories. This type of data is common in social science research settings where the dependent variable often comes from opinion polls or evaluations. For example, ordinal regression can be used to predict the letter grades of students based on the time they spend studying, or Likert scale responses to a survey based on the annual income of the respondent.

In People Analytics at Shopify, we use ordinal regression to empower Shopify employees. Our annual engagement survey contains dozens of scale questions about wellness, team health, leadership and alignment. To better dig into this data we built `bevel`, a repository that contains simple, easy-to-use Python implementations of standard ordinal regression techniques.

#### Using bevel

```python
from bevel import OrdinalRegression

orf = OrdinalRegression()
orf.fit(X, y)
orf.print_summary()

orf.predict_class(X)
orf.predict_probabilities(X)
```

