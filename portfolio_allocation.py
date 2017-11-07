### Portfolio Allocation

# Finds an optimal allocation of stocks in a portfolio,
# satisfying a minimum expected return.
# The problem is posed as a Quadratic Program, and solved
# using the cvxopt library.
# Uses actual past stock data, obtained using the stocks module.

from cvxopt import matrix, solvers
import stocks
import numpy

# solves the QP, where x is the allocation of the portfolio:
# minimize   x'Px + q'x
# subject to Gx <= h
#            Ax == b
#
# Input:  n       - # of assets
#         avg_ret - nx1 matrix of average returns
#         covs    - nxn matrix of return covariance
#         r_min   - the minimum expected return that you'd
#                   like to achieve
# Output: sol - cvxopt solution object
def optimize_portfolio(n, avg_ret, covs, r_min):
	P = covs
	# x = variable(n)
	q = matrix(numpy.zeros((n, 1)), tc='d')
	# inequality constraints Gx <= h
	# captures the constraints (avg_ret'x >= r_min) and (x >= 0)
	G = matrix(numpy.concatenate((
		-numpy.transpose(numpy.array(avg_ret)),
		-numpy.identity(n)), 0))
	h = matrix(numpy.concatenate((
		-numpy.ones((1,1))*r_min,
		numpy.zeros((n,1))), 0))
	# equality constraint Ax = b; captures the constraint sum(x) == 1
	A = matrix(1.0, (1,n))
	b = matrix(1.0)
	sol = solvers.qp(P, q, G, h, A, b)
	return sol

### setup the parameters
symbols = ['GOOG', 'AIMC', 'CE', 'BH', 'AHGP', 'AB', 'HLS', 'BKH', 'LUV']
# pull data from this date range
start   = '1/1/2010'
end     = '1/1/2016'
n       = len(symbols)
# average yearly return for each stock
avg_ret = matrix(map(lambda s: stocks.avg_return(s, start, end, 'y'), symbols))
# covariance of asset returns
covs    = matrix(numpy.array(stocks.cov_matrix(symbols, start, end, 'y')))
# minimum expected return threshold
r_min   = 0.10

### solve
solution = optimize_portfolio(n, avg_ret, covs, r_min)

print solution['x']
