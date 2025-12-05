| Model Type | from | import |
|-----------|------|--------|
| Linear Models (Regression & Classification) | `from sklearn.linear_model` | `LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet, SGDRegressor, SGDClassifier, PassiveAggressiveClassifier, PassiveAggressiveRegressor, RidgeClassifier` |
| Support Vector Machines (SVM) | `from sklearn.svm` | `SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR` |
| Neighbors (KNN family) | `from sklearn.neighbors` | `KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor, NearestNeighbors` |
| Decision Tree | `from sklearn.tree` | `DecisionTreeClassifier, DecisionTreeRegressor` |
| Ensemble Models | `from sklearn.ensemble` | `RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier, BaggingRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor` |
| Naive Bayes | `from sklearn.naive_bayes` | `GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB` |
| Discriminant Analysis | `from sklearn.discriminant_analysis` | `LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis` |
| Clustering | `from sklearn.cluster` | `KMeans, MiniBatchKMeans, DBSCAN, OPTICS, MeanShift, AgglomerativeClustering, SpectralClustering` |
| Mixture Models | `from sklearn.mixture` | `GaussianMixture, BayesianGaussianMixture` |
| Dimensionality Reduction | `from sklearn.decomposition` | `PCA, KernelPCA, SparsePCA, TruncatedSVD, NMF` |
| Manifold Learning | `from sklearn.manifold` | `TSNE, Isomap, LocallyLinearEmbedding` |
| Feature Selection | `from sklearn.feature_selection` | `SelectKBest, SelectPercentile, RFE, RFECV, VarianceThreshold, SelectFromModel` |
| Preprocessing | `from sklearn.preprocessing` | `StandardScaler, MinMaxScaler, RobustScaler, Normalizer, OneHotEncoder, LabelEncoder, OrdinalEncoder, PolynomialFeatures` |
| Model Selection (Training Tools) | `from sklearn.model_selection` | `train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold` |
| Pipeline | `from sklearn.pipeline` | `Pipeline` |
| Metrics | `from sklearn.metrics` | `accuracy_score, mean_squared_error, confusion_matrix, classification_report` |
| External Libraries | `import` | `xgboost as xgb, from lightgbm import LGBMClassifier, LGBMRegressor, from catboost import CatBoostClassifier, CatBoostRegressor` |
