# Product Classification with Keras

#### Business Case
To be able to recognise which catergory a product belongs based on set of event counts. Products, events and categories are anonymised.

#### Dataset
Otto product classification (https://www.kaggle.com/c/otto-group-product-classification-challenge)
* 9 product categories.
* Counts of 93 anonymous events.
* Almost 62,000 products.

#### Approach
To see how well a pretty standard network performs with this dataset, using `relu` and `softmax` activations, compiled with `rmsprop` optimisation. The nodes will gradually decrease in number over 8 hidden layers.

1. Preprocess with `pandas`, `numpy`, `sklearn`.
2. Build and fit model using `Keras`.
3. Evaluate with Kfold cross-validation with `sklearn`.

#### Performance:
* =<78% accuracy.
* A large amount of overfitting.

There are several possiblities for overfitting here. A high number of nodes were picked to spot complex relationships, which would cause overfitting if the complexity is less than expected. Alternatively, the `relu` activation function has also been known to overfit despite being one of the most used methods, and of course there is some potential for odd cases in the data.

#### Environment
* Install dependencies using `pip` and the command `pip install -r requirements.txt`
