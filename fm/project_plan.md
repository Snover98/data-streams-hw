# AMS Plan
## How to generate data
Take a distribution (maybe log normal) with very high variance and sample it (and round the values).

To make the hash function, just take a list of the values and shuffle them - make $\dfrac{1}{index(val)}$ the hash value,
and maybe put it in a dictionary so that the lookups won't be that long.

Using this we can init the different $\alpha$-estimators with different hash functions, and the $\beta$-estimators will have their own lists of alpha estimators.

Using array operators this could be done parallely on large chunks for all estimators.\
Given the limited CPU & RAM of this laptop it would probably be best to try and optimize the runtime of a single experiment and just run them inro order.

# Counter Plan
## Choose an algo
Need to choose between the 2 algos first :(