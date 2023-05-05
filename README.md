# LearningCookbooks
### Within Venv: Python 3.11.3 - MacOS Ventura 13.3.1 - Macbook Air M2

External link: [Create a Virtual Environment with Python](https://gist.github.com/loic-nazaries/c25ce9f7b01b107573796b026522a3ad#file-create_a_virtual_environment_with_python-md)

In VsCode: Cmd+Shift+P, Create Environment, Venv.

To install required packages from [requirements.txt](requirements.txt), run:

    pip install -r requirements.txt

## File Execution Time
##### plots generated normally while plt.show() supressed for execution time calculations

| File Name | Execution Time |
| --- | --- |
| 1NumPyIntro.py | 0.17 seconds |
| 2LoadingData.py | 1.52 seconds |
| 3DataWrangling.py | 1.26 seconds |
| 4NumericalData.py | 0.79 seconds |
| 5CategoricalData.py | 0.72 seconds |
| 6Text.py | 1.58 seconds |
| 7DateTime.py | 0.40 seconds |
| 8Images.py | 1.03 seconds |
| 9DimensionalityReductionFeatureExtraction.py | 8.48 seconds |
| 10DimensionalityReductionFeatureSelection.py | 0.67 seconds |
| 11ModelValuation.py | 26.97 seconds |

1. Introduction to Numpy Arrays, Matrixes, and Scipy Sparse Arrays. 
    1. Vectorizing lambda and other functions.
    2. Finding extremes, mean variance, standard deviation, and means.
    3. Reshaping, transposing, and flattening.
    4. Determinant, tracing, diagonals, and sums.
    5. Eigenvalues and eigenvectors.
    6. Mul.Div.Add.Sub. of matrices. Dot Prod & Inverses.
    7. Np Randomization.
2. Loading Data
    1. Loading data, processing data.
    2. Creating regression (for linear regression).
    3. Making classification (for classification).
    4. Making blobs (for clustering).
    5. Reading csv, excel, and json with Pandas.
3. Data Wrangling (Using Pandas Dataframes).
    1. Creating dataframes, columns, and rows.
    2. Concatenating.
    3. Info from head, shape, and describe.
    4. Selecting rows and setting index.
    5. Replacing and renaming.
    6. Statistics using max, min, mean, sum, count.
    7. Values counts, unique, and nunique.
    8. Dropping duplicates, resampling, and sorting.
4. Numerical Data
    1. Creating scaled features
    2. Creating standard features
    3. Creating robust features
    4. Creating normalized features
    5. Creating polynomial features
    6. Using a function transformer
    7. Making detector, fitter, predictor
    8. Creating binarized features
    9. Clustering using Kmeans, fitting, and predicting
5. Categorical Data
    1. Multiclass binarization
    2. Mapping features
    3. Pd vs dict mapping
    4. Predicting values
    5. Random forest classifier
6. Text
    1. Cleaning text
    2. Tokenizing text
    3. Tokenizing sentences
    4. Removing stop words
    5. Root Stemming words
    6. Tagging speech
    7. Encode text as word bags
    8. Word weighting
7. DateTime
    1. Converting strings to datetimes
    2. Setting and converting time zones
    3. Selecting dates and times
    4. Date data to features
    5. Creating a lag feature
    6. Using rolling time windows
    7. Handling missing values
8. Images
    1. Loading images
    2. Converting images to grayscale
    3. Resizing images
    4. Flattening images
    5. Plotting images
    6. Plotting image histograms
    7. Plotting image arrays
9. Dimensionality Reduction and Feature Extraction
    1. Principal component reduction
    2. Linearly inseparable reduction
    3. Maximize class separation
    4. Matrix factorization reduction
    5. Sparse data reduction
10. Dimensionality Reduction and Feature Selection
    1. Threshold numerical feature variance
    2. Threshold binary feature variance
    3. Highly correlated features
    4. Classification by irrelevant feature removal
    5. Recursive feature elimination
11. Model Valuation
    1. Cross validation
    2. Baseline regression model
    3. Baseline classification model
    4. Evaluate binary classifier predictions
    5. Evaluate binary classifier thresholds
    6. Evaluate regression models
    7. Evaluate clustering models
    8. Visualize training set
    9. Text report evaluation metrics
    10. Visualize hyperparameter values