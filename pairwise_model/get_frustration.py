"""Get the frustrated states. Written by Markus Ekvall 2018-07-09."""
import numpy as np
from scipy.special import comb


def get_frustration(J, verbose=False):
    """
    Get the fractions of frustrated states.

    ----------
    J : array
       The interactions
    verbose : bool
       Print the progress

    """
    [m, n] = np.shape(J)
    nr_iteration = 2*comb(m, 3)
    ix_over = np.triu_indices(m, k=1)
    ix_under = np.tril_indices(m, k=-1)
    upper_J = J[ix_over].copy()
    lower_J = J[ix_under].copy()
    harmony = 0
    frustration = 0
    limit = 0.01

    for i in range(0, len(lower_J) - 1):
        lower_row = ix_under[0][i]
        lower_col = ix_under[1][i]
        for k, (upper_row, upper_col) in enumerate(zip(ix_over[0],
                                                       ix_over[1])):
            if lower_col == upper_row and upper_col != lower_row:
                cond1 = np.sign(lower_J[i]) == np.sign(upper_J[k])
                cond2 = np.sign(upper_J[k]) == np.sign(lower_J[k])
                cond3 = np.sign(lower_J[k]) == np.sign(lower_J[i])
                if cond1 and cond2 and cond3:
                    harmony += 1
                    if verbose:
                        print("|Hamronic traingle|: ",
                              (lower_row, lower_col), "-->",
                              (upper_row, upper_col),
                              "-->", (upper_col, lower_row))
                else:
                    frustration += 1
                    if verbose:
                        print("|Frustrated traingle|: ",
                              (lower_row, lower_col), "-->",
                              (upper_row, upper_col),
                              "-->", (upper_col, lower_row))
            if limit <= (frustration+harmony)/nr_iteration:
                print("||Progress||: "+str(int(limit*100))+" %")
                limit = (frustration+harmony)/nr_iteration+0.01
    print("||Progress||: 100%")
    harmony = harmony/float(frustration+harmony)
    print("||Frustrated state||: "+str((1-harmony)*100)+"%")
    print("||Harmonic state||: "+str(harmony*100)+"%")
    return harmony
