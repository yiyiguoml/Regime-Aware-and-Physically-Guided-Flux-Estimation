from math import log, atan, sqrt
import numpy as np

# @jit(nopython=True)
def mo_similarity_two_levels(u_low, v_low, u_high, v_high, t_low, t_high, pressure,
                             z_low, z_high, mixing_ratio_low, mixing_ratio_high, z0=0.01):
    """
    Calculate flux information based on Monin-Obukhov similarity theory with instruments at two levels.

    Args:
        u_low: u at lower level in m/s
        v_low: v at lower level in m/s
        u_high: u at higher level in m/s
        v_high: v at higher level in m/s
        t_low: temperature at lower level (K)
        t_high: temperature at higher level (K)
        pressure: surface pressure in hPa
        z_low: height of lower level in m
        z_high: height of higher level in m
        z0: momentum roughness length
        mixing_ratio_low: mixing ratio at lower level in g/kg   
        mixing_ratio_high: mixing ratio at higher level in g/kg

    Returns:
        ustar: friction velocity m/s
        tstar: temperature scale K
        qstar: moisture scale [g/kg]
        wthv0: kinematic sensible heat flux proxy [K m/s]
        wqv0: kinematic moisture flux proxy [g/kg m/s]
        zeta_high: stability parameter z/L at z_high [-]
        phi_m: momentum universal stability function [-]
        phi_h: heat universal stability function [-]
    """
    z_ratio = (z_high - z0) / (z_low - z0)
    # else:
    #    sys.exit("Surface roughnes, z0, must be greter than 0.!")
    #
    # Gravitational acceleration
    g = 9.81
    #
    # Gas constant over spcific heat capacity at constant pressure
    r = 287.058
    cp = 1005.
    rocp = r / cp
    #
    # Reference pressure and temperature
    p0 = 1000.
    t0 = 300.
    #
    # Set M-O parameters based on Dyer 1974 paper
    karman = 0.4
    beta = 5.0
    gamma = 16.0
    #
    # Potential temperature speed at level 1
    th_high = t_high * (p0 / pressure) ** rocp
    th_low = t_low * (p0 / pressure) ** rocp
    #
    # Small number
    epsilon = 1.e-6
    #
    # Initial values of drag coefficients - neutrally stratified case
    cd = karman ** 2 / ((log(z_ratio)) ** 2)
    ch = karman ** 2 / ((log(z_ratio)) ** 2)
    cq = karman ** 2 / ((log(z_ratio)) ** 2)
    #
    # Initial values of surface friction velocity, temperature scale, and
    # heat flux
    wind_speed_high = sqrt(u_high * u_high + v_high * v_high)
    wind_speed_low = sqrt(u_low * u_low + v_low * v_low)
    if wind_speed_high < 0.1:
        wind_speed_high = 0.1

    if wind_speed_low < 0.01:
        wind_speed_low = 0.01
    ustar = (cd * (wind_speed_high - wind_speed_low) ** 2) ** 0.5
    if ustar < 0.01:
        ustar = 0.01

    tstar = -ch / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (th_low - th_high)
    wthv0 = -ustar * tstar

    qstar = cq / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (mixing_ratio_low - mixing_ratio_high)
    wqv0 = -ustar * qstar
    #
    # Set stopping criterion
    diff = 1.
    #
    # Set stability functions
    psi_m = 0.
    psi_m_low = 0.
    psi_h = 0.
    psi_h_low = 0.
    phi_m = 0.
    phi_h = 0.
    #
    zeta_high = 0.
    zeta_low = 0.
    count = 0
    while diff > epsilon and count < 100:
        #
        # Surface friction velocity and temperature scale
        ustar = (cd * (wind_speed_high - wind_speed_low) ** 2) ** 0.5
        if ustar < 0.01:
            ustar = 0.01
        tstar = -ch / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (th_low - th_high)
        wthv0 = -ustar * tstar

        qstar = cq / ustar * ((wind_speed_high - wind_speed_low) ** 2) ** 0.5 * (mixing_ratio_low - mixing_ratio_high)
        wqv0 = -ustar * qstar

        #
        # Compute drag coefficients
        cdold = cd
        chold = ch
        cqold = cq

        #
        # Neutrally stratified case
        if wthv0 == 0:
            zeta_high = 0.
            zeta_low = 0.
            psi_m = 0.
            psi_m_low = 0.
            psi_h = 0.
            psi_h_low = 0.
            phi_m = 1.
            phi_h = 1.
            cd = karman ** 2 / ((log(z_ratio)) ** 2)
            ch = karman ** 2 / ((log(z_ratio)) ** 2)
            cq = karman ** 2 / ((log(z_ratio)) ** 2)
        elif abs(wthv0) > 0:
            #
            # Obukhov length scale
            olength = -ustar ** 3 / (karman * g / th_high * wthv0)
            if abs(olength) < (z_high) and olength > 0:
                olength = z_high
            elif abs(olength) < z_high and olength < 0:
                olength = -(z_high)
            #
            # Free convection
            # Monin-Obukhov stability parameter
            zeta_high = z_high / olength
            zeta_low = z_low / olength
            #
            # Convective case
            if (zeta_high >= -2.) & (zeta_high < -epsilon):
                xi_high = 1. / ((1. - gamma * zeta_high) ** 0.25)
                xi_low = 1. / ((1. - gamma * zeta_low) ** 0.25)

                psi_m = log(0.5 * (1.0 + xi_high ** 2) * (0.5 * (1.0 + xi_high)) ** 2) \
                        - 2. * atan(xi_high) + 0.5 * np.pi
                psi_m_low = log(0.5 * (1.0 + xi_low ** 2) * (0.5 * (1.0 + xi_low)) ** 2) \
                            - 2. * atan(xi_low) + 0.5 * np.pi

                psi_h = 2.0 * log(0.5 * (1.0 + xi_high ** 2))
                psi_h_low = 2.0 * log(0.5 * (1.0 + xi_low ** 2))

                psi_q = psi_h
                psi_q_low = psi_h_low

                phi_m = 1. / ((1. - gamma * zeta_high) ** 0.25)
                phi_h = 1. / ((1. - gamma * zeta_high) ** 0.25)
                phi_q = 1. / ((1. - gamma * zeta_high) ** 0.25)
            #
            # Stably stratified case
            elif (zeta_high > epsilon) & (zeta_high <= 1.):
                psi_m = - beta * zeta_high
                psi_h = - beta * zeta_high
                psi_m_low = -beta * zeta_low
                psi_h_low = -beta * zeta_low
                psi_q = psi_h
                psi_q_low = psi_h_low

                phi_m = (1. + beta * zeta_high)
                phi_h = (1. + beta * zeta_high)
                phi_q = phi_h
            #
            # Neutrally stratified case
            elif (zeta_high <= epsilon) & (zeta_high >= -epsilon):
                psi_m = 0.
                psi_h = 0.
                psi_m_low = 0.
                psi_h_low = 0.
                psi_q = psi_h
                psi_q_low = psi_h_low

                phi_m = 1.
                phi_h = 1.
                phi_q = 1.
            #
            cd = karman ** 2 / ((log(z_ratio) - psi_m + psi_m_low) ** 2)
            ch = karman ** 2 / ((log(z_ratio) - psi_m + psi_m_low) * (log(z_ratio) - psi_h + psi_h_low))
            cq = karman ** 2 / ((log(z_ratio) - psi_m + psi_m_low) * (log(z_ratio) - psi_q + psi_q_low))
        #
        diff = abs(cd - cdold) + abs(ch - chold) + abs(cq - cqold)
        count += 1
    #
    return ustar, tstar, qstar, wthv0, wqv0, zeta_high, phi_m, phi_h

# wrapper
def mo_similarity_two_levels_vec(u_low, v_low, u_high, v_high,
                                 t_low_k, t_high_k, p_high_hpa,
                                 z_low, z_high,
                                 mr_low, mr_high,
                                 z0):
    """
    Vector wrapper for mo_similarity_two_levels (scalar).
    Returns an array of shape (n, 8).
    """

    u_low  = np.asarray(u_low, dtype=np.float64).ravel()
    v_low  = np.asarray(v_low, dtype=np.float64).ravel()
    u_high = np.asarray(u_high, dtype=np.float64).ravel()
    v_high = np.asarray(v_high, dtype=np.float64).ravel()

    t_low_k  = np.asarray(t_low_k, dtype=np.float64).ravel()
    t_high_k = np.asarray(t_high_k, dtype=np.float64).ravel()
    p_high_hpa = np.asarray(p_high_hpa, dtype=np.float64).ravel()

    mr_low  = np.asarray(mr_low, dtype=np.float64).ravel()
    mr_high = np.asarray(mr_high, dtype=np.float64).ravel()

    n = u_low.size
    out = np.empty((n, 8), dtype=np.float64)

    for i in range(n):
        out[i, :] = mo_similarity_two_levels(
            float(u_low[i]), float(v_low[i]), float(u_high[i]), float(v_high[i]),
            float(t_low_k[i]), float(t_high_k[i]), float(p_high_hpa[i]),
            float(z_low), float(z_high),
            float(mr_low[i]), float(mr_high[i]),
            z0=float(z0)
        )

    return out