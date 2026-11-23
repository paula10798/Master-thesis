from thesis.models.mixtures import (main_tt, main_ttt, main_sN_t,
                                    main_sN_t_n, main_nn)
from thesis.models.mar1_mixtures import (main_mar1_nn, main_mar1_tt,
                                         main_mar1_sN_t, main_mar1_ttt)

## Other Mixture models and MAR (1) considered ##

# Note.... take out the '#' before main_ to print the results for each model
print('Results from other Mixture and MAR (1) models....')

# Mixture Model with two Student-t distributions
#main_tt()

# Mixture Model with three Student-t distributions
#main_ttt()

# Mixture Model with Skewed Normal and Student-t distributions
#main_sN_t()

# Mixture model with Student t, Skewed Normal and Normal distributions
#main_sN_t_n()

# Mixture Model with two Gaussian distributions
#main_nn()

# MAR(1) Model with two Gaussian distributions
#main_mar1_nn()

# MAR (1) Model with two Student-t distributions
#main_mar1_tt()

# MAR (1) Model with Skewed Gaussian and Student-t distributions
#main_mar1_sN_t()

# MAR(1) Model with three Student-t distributions
main_mar1_ttt()
