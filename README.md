# Regularization-methods


#####Data source:

1. Boston dataset

CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63) ^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's


2. Energy dataset

date time year-month-day hour: minute: second	T7, Temperature in ironing room, in Celsius
lights, energy use of light fixtures in the house in Watts per hour	RH_7, Humidity in ironing room, in %
T1, Temperature in kitchen area, in Celsius	T8, Temperature in teenager room 2, in Celsius
RH_1, Humidity in kitchen area, in %	RH_8, Humidity in teenager room 2, in %
T2, Temperature in living room area, in Celsius	T9, Temperature in parents room, in Celsius
RH_2, Humidity in living room area, in %	RH_9, Humidity in parents room, in %
T3, Temperature in laundry room area	To, Temperature outside (from Chievres weather station), in Celsius
RH_3, Humidity in laundry room area, in %	Pressure (from Chievres weather station), in mm Hg
T4, Temperature in office room, in Celsius	RH_out, Humidity outside (from Chievres weather station), in %
RH_4, Humidity in office room, in %	Wind speed (from Chievres weather station), in m/s
T5, Temperature in bathroom, in Celsius	Visibility (from Chievres weather station), in km
RH_5, Humidity in bathroom, in %	Tdewpoint (from Chievres weather station), Â°C
T6, Temperature outside the building (north side), in Celsius	rv1, Random variable 1, non-dimensional
RH_6, Humidity outside the building (north side), in %	rv2, Random variable 2, non-dimensional
Attribute Information (target):    Appliances, energy use in Watts per hour






######Models:

1. Multiple linear regression
2. MLP
3. SVM

MLR	
  MLR: Multiple Linear Regression
	Ridge: Ridge Regression
	Ridge.CV: Ridge Regression with Cross Validation
	Lasso: Lasso Regression
	Lasso.CV: Lasso Regression with Cross Validation
	ElasticNet: Elastic net Regression
	ElasticNet.CV: Elastic net Regression with Cross Validation

SVM	
  SVM: Support Vector Machine (C=0.5)
	SVM: Support Vector Machine (C=1)
	SVM.ℓ1: Support Vector Machine with ℓ1(Lasso) penalty
	SVM.ℓ2: Support Vector Machine with ℓ2(Ridge) penalty
  
MLP	
  MLP: Multilayer Perceptron
	MLP.ℓ1: Multilayer Perceptron with ℓ1(Lasso) penalty
	MLP.ℓ2: Multilayer Perceptron with ℓ2(Ridge) penalty
	MLP.Elastic: Multilayer Perceptron with elastic net






#########Regularization methods

l1, l2, ElasticNet






