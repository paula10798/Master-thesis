from thesis.models.mixtures import main_n_t
from thesis.models.mar1_mixtures import main_mar1_nt

## Modelling Non-trading period returns ##

print('We will now attempt to model the log-returns over the Weekend and Overnight periods')
print('----------------------------------')

###################
# Weekend Returns #
###################

print('WEEKEND LOG-RETURNS...')
print('NOTE... I have included for the best performing model below. While the code for the models can be found towards the bottom')
print('---------------------')

print('Mixture Model with Student-t and Gaussian distribution')
## Mixed model with Student-t and Gaussian distributions ##

main_n_t()

print('-------------------------------------')

#####################
# Overnight Returns #
#####################

print('OVERNIGHT LOG-RETURNS...')
print('NOTE... I have included for the best performing model below. While the code for the models can again be found towards the bottom')
print('---------------------')

print('Mixture Autoregressive Model (1) with Student-t and Gaussian distribution')

## MAR (1) with Student-t and Gaussian mixed distribution ##

main_mar1_nt()

print('-------------------------------------')
