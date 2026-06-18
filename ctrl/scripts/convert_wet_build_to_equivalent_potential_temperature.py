import numpy as np

def wet_bulb_to_equivalent_potential_temperature_claude(theta_w, p0=1000.0):
    """
    Claude's attempt.
    Convert wet-bulb potential temperature theta_w [K] to pseudo-equivalent
    potential temperature theta_e [K] via Bolton (1980), with no iteration.

    A parcel on a given pseudoadiabat is, by definition, saturated at p0 = 1000 hPa
    with temperature theta_w. Evaluating Bolton's theta_e expression for that
    saturated 1000-hPa parcel therefore returns theta_e of the same adiabat as a
    closed-form function of theta_w alone.

    Bolton (1980) pieces used:
      - saturation vapour pressure  e_s(T)            (Eq. 10)
      - LCL temperature T_L; here T_L = theta_w       (Eq. 15, since already saturated at p0)
      - dry potential temperature at the LCL          (Eq. 24)
      - equivalent potential temperature              (Eq. 43, the accurate empirical form)

    Parameters
    ----------
    theta_w : array_like
        Wet-bulb potential temperature [K] (scalar or any-shape array).
    p0 : float
        Reference pressure [hPa]; 1000 by definition of theta_w.

    Returns
    -------
    ndarray
        Equivalent potential temperature [K], same shape as theta_w.
    """
    Tw = np.asarray(theta_w, dtype=float)            # K
    # Can you spot the mistake? Tc is meant to be temperature in Celsius. Here, it's
    # theta_w - 273.15.
    Tc = Tw - 273.15                                 # degC

    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))   # Eq. 10, hPa
    r  = 622.0 * es / (p0 - es)                       # saturation mixing ratio, g/kg
    # Not sure about these steps.
    T_L = Tw                                          # Eq. 15 collapses to theta_w
    theta_dl = Tw * (p0 / (p0 - es)) ** 0.2854        # Eq. 24 ((T/T_L) factor = 1 here)
    theta_e = theta_dl * np.exp((3.376 / T_L - 0.00254) * r * (1.0 + 0.81e-3 * r))  # Eq. 43
    return theta_e


def wet_bulb_to_equivalent_potential_temperature(T, theta_w, p0=1000.0):
    """
    Mark's
    Convert wet-bulb potential temperature theta_w [K] to pseudo-equivalent
    potential temperature theta_e [K] via Bolton (1980), with no iteration.


    Returns
    -------
    ndarray
        Equivalent potential temperature [K], same shape as theta_w.
    """
    theta_w = np.asarray(theta_w, dtype=float)            # K
    Tc = T - 273.15                                       # degC

    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))   # Eq. 10, hPa
    rs  = 622.0 * es / (p0 - es)                     # Eq. 41, saturation mixing ratio, g/kg
    theta_e = theta_w * np.exp(((3.376 / theta_w) - 0.00254) * rs * (1 + 0.81e-3 * rs))
    return theta_e


if __name__ == '__main__':
    # --- quick self-test ---
    for tw_c in [16, 18, 20, 22, 24]:
        tw = tw_c + 273.15
        te = wet_bulb_to_equivalent_potential_temperature(tw)
        print(f"theta_w = {tw_c:2d} degC ({tw:6.2f} K)  ->  theta_e = {te:7.2f} K")

    grid = np.array([[288.0, 293.0], [296.0, 299.0]])
    print("\nvectorised:\n", wet_bulb_to_equivalent_potential_temperature(grid))
