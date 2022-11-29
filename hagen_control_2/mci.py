import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Example 01 
a = 0
b = 5
x = np.random.uniform(low=a, high=b, size=1000)


def integrand(x):
    return x ** 2


# Monte Carlo integration
y = integrand(x)
y_bar = ((b - a) / x.shape[0]) * np.sum(y)

I = quad(integrand, a, b, )
print("Actual integration: ", I[0], "Monte Carlo integration: ", y_bar)
plt.scatter(x, y)
plt.show()

# Example 02
a = 0
b = 1
x = np.random.uniform(low=a, high=b, size=1000)


def integrand2(x):
    return 4 * np.sqrt((1 - (x ** 2)))


# Monte Carlo integration
y = integrand2(x)
y_bar = ((b - a) / x.shape[0]) * np.sum(y)

I = quad(integrand2, a, b, )
print("Actual integration: ", I[0], "Monte Carlo integration: ", y_bar)
plt.scatter(x, y)
plt.show()
