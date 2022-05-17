import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd


n_samples = 500  # number of samples
noise = 1  # bias and standard deviation of the guassian noise


x, y = datasets.make_regression(
    n_samples=n_samples,  # number of samples
    n_features=1,  # number of features
    n_informative=1,  # number of useful features
    noise=noise,  # bias and standard deviation of the guassian noise
    random_state=10,  # random seed
)  # set for same data points for each run

# Scale Y axis (YoX) to 0.1 to 20 Years to 2DP
y = np.around(np.interp(y, (y.min(), y.max()), (0.1, 20)), decimals=2)

# Scale X axis (salary) to range 20000..150000
x = np.around(np.interp(x, (x.min(), x.max()), (20000, 150000)), decimals=2)

plt.plot(x, y, ".", label="training data")
plt.xlabel("Salary (£)")
plt.ylabel("Years of experience Y")
plt.title("Experience Vs. Salary")
plt.show()

# Create a dataframe with the x and y values
df = pd.DataFrame({"YearsExperience": y, "Salary": x.flatten()})
df.to_csv("./LinearRegression/Data/Custom_Salary_Data.csv", index=False)
