# https://www.gaussianwaves.com/2020/01/generating-simulated-dataset-for-regression-problems-sklearn-make_regression/
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt  # for plotting
import pandas as pd

x, y = datasets.make_regression(
    n_samples=500,  # number of samples
    n_features=1,  # number of features
    n_informative=1,  # number of useful features
    noise=5,  # bias and standard deviation of the guassian noise
    random_state=61,  # random seed
)  # set for same data points for each run

# Scale feature x (years of experience) to range 0..20
y = np.around(np.interp(y, (y.min(), y.max()), (0.1, 20)), decimals=2)

# Scale target y (salary) to range 20000..150000
x = np.around(np.interp(x, (x.min(), x.max()), (20000, 150000)), decimals=2)

plt.plot(x, y, ".", label="training data")
plt.xlabel("Salary")
plt.ylabel(" Years of experience $")
plt.title("Experience Vs. Salary")
plt.show()


# Create a dataframe with the x and y values
df = pd.DataFrame({"YearsExperience": y, "Salary": x.flatten()})
df.to_csv("./LinearRegression/Data/Custom_Salary_Data.csv", index=False)
