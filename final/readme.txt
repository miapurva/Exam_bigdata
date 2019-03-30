1. Establish association between attributes and stars rating.
	--> python arm.py
2. Fit any two classifiers for each stars rating.
	RandomForest classifier and GradientBoost Classifier is used
	-->python 1RF.py
	-->python 1GB.py
3. Cluster customer reviews using similarity metric.
	Kmeans clustering is used
	-->python kmeans.py
4. Evaluate the quality of clustering in (3).
	Purity metric is used to evaluate the quality.
	--> python purity.py