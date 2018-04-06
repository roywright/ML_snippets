# ML snippets
Some of the code I've put together to do weird/interesting/cool things (mostly at work) with machine learning models for classification problems. The core code is in `ML_snippets.py` and usage examples are shown in `demos.ipynb`. This is heavily reliant on the pandas and scikit-learn libraries, though it wouldn't be too difficult to rewrite and avoid the need for pandas. 

`explanations_demo.ipynb` has a demonstration of two methods for generating human-interpretable explanations/justifications of the predictions of a scikit-learn random forest classifier. The code for one of those methods is found in `tree_explainer.py`.

`price_clustering.ipynb` has a demo of segmenting a set of products into natural price bins. That notebook is self-contained; it pulls a small data set from the web.
