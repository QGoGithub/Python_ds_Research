 ##One of the key trading concepts in the quantitative toolbox is that of mean reversion.
 ##This process refers to a time series that displays a tendency to revert to its historical mean value.
 ##Mathematically, such a (continuous) time series is referred to as an Ornstein-Uhlenbeck process.
 ##This is in contrast to a random walk (Brownian motion), which has no "memory" of where it has been at
 ##each particular instance of time. The mean-reverting property of a time series can be exploited in
 ##order to produce profitable trading strategies.

## We must carry out statistical tests to identify mean reversion.
## The first step is to test for stationarity

##Testing for Mean Reversion
##A continuous mean-reverting time series can be represented
##by an Ornstein-Uhlenbeck stochastic differential equation:

##  dxt=θ(μ−xt)dt+σdWt

## xt is the price of the asset under investigation at time period t
##θ is the rate of reversion to the mean,
## μ is the mean value of the process,
## σ is the variance of the process
## Wt is a Wiener Process or Brownian Motion.

##In a discrete setting the equation states that the change of the price series
##in the next time period is proportional to the difference between the mean
##price and the current price, with the addition of Gaussian noise.

## Test for stationarity STEP 1  - Augmented Dickey-Fuller Test

## We test for the presence of a unit root in an autoregressive time series sample.
## Intuitively we know that that if a price series possesses mean reversion,
## then the next price level will be proportional to the current price level.
## A linear lag model of order p is used for the time series:

## Δyt=α+βt+γyt−1+δ1Δyt−1+⋯+δp−1Δyt−p+1+ϵt


# α is a constant,
# β represents the coefficient of a temporal trend
# Δyt=y(t)−y(t−1)Δyt=y(t)−y(t−1).

## The ADF hypothesis test checks against the null hypothesis that γ=0,
## ie.  α=β=0α=β=0 - that the process is a random walk and thus non mean reverting.

##If the hypothesis that γ=0 can be rejected then the following
##movement of the price series is proportional to the current price and thus it is unlikely to be a random walk.

## Data Series:
## Google price series from 2000-01-01 to 2013-01-01
## Google price series from 2000-01-01 to 2013-01-01

## ADF TEST ##

# Import the Time Series library
import statsmodels.tsa.stattools as ts

# Import Datetime and the Pandas DataReader
from datetime import datetime
from pandas.io.data import DataReader

# Download the Google OHLCV data from 1/1/2000 to 1/1/2013
goog = DataReader("GOOG", "yahoo", datetime(2000,1,1), datetime(2013,1,1))

# Output the results of the Augmented Dickey-Fuller test for Google
# with a lag order value of 1
ts.adfuller(goog['Adj Close'], 1)
#Here is the output of the Augmented Dickey-Fuller test for Google over the period.
#The first value is the calculated test-statistic, while the second value is the p-value.
#The fourth is the number of data points in the sample. The fifth value, the dictionary,
#contains the critical values of the test-statistic at the 1, 5 and 10 percent values respectively.

'''(-2.1900105430326064,
 0.20989101040060731,
 0,
 2106,
 {'1%': -3.4334588739173006,
  '10%': -2.5675011176676956,
  '5%': -2.8629133710702983},
 15436.871010333041)
'''
 ##Since the calculated value of the test statistic is larger than any of the critical
 ##values at the 1, 5 or 10 percent levels, we cannot reject the null hypothesis
 ##and thus we are unlikely to have found a mean reverting time series.

##An alternative means of identifying a mean reverting time series is provided by the concept of stationarity,

##Testing for Stationarity STEP 2 - HURST EXPONENT

##A time series (or stochastic process) is defined to be strongly stationary if
##its joint probability distribution is invariant under translations in time or space.
##In particular, and of key importance for traders, the mean and variance of the process
##do not change over time or space and they each do not follow a trend.

##A critical feature of stationary price series is that the prices within the series
##diffuse from their initial value at a rate slower than that of a Geometric Brownian Motion.
##By measuring the rate of this diffusive behaviour we can identify the nature of the time series.

##We will use the Hurst Exponent, which helps us to characterise the stationarity of a time series.


from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn

def hurst(ts):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, 100)

	# Calculate the array of the variances of the lagged differences
	tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

	# Use a linear fit to estimate the Hurst Exponent
	poly = polyfit(log(lags), log(tau), 1)

	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

# Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
gbm = log(cumsum(randn(100000))+1000)
mr = log(randn(100000)+1000)
tr = log(cumsum(randn(100000)+1)+1000)

# Output the Hurst Exponent for each of the above series
# and the price of Google (the Adjusted Close price) for
# the ADF test given above in the article
print "Hurst(GBM):   %s" % hurst(gbm)
print "Hurst(MR):    %s" % hurst(mr)
print "Hurst(TR):    %s" % hurst(tr)

# Assuming you have run the above code to obtain 'goog'!
print "Hurst(GOOG):  %s" % hurst(goog['Adj Close'])
#The output from the Hurst Exponent Python code is given below:

'''Hurst(GBM):   0.500606209426
Hurst(MR):    0.000313348900533
Hurst(TR):    0.947502376783
Hurst(GOOG):  0.507880122614
'''

'''From this output we can see that the Geometric Brownian Motion posssesses a Hurst Exponent,
HH, that is almost exactly 0.5. The mean reverting series has HH almost equal to zero,
while the trending series has HH close to 1.

Interestingly, Google has HH also nearly equal to 0.5 indicating that it is
extremely close to a geometric random walk (at least for the sample period we're making use of).

While we now have a means of characterising the nature of a price time series,
we have yet to discuss how statistically significant this value of HH is.
We need to be able to determine if we can reject the null hypothesis that H=0.5H=0.5 to
 ascertain mean reverting or trending behaviour.
'''
