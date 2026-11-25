def mortality_R(T, m_min_R, c_R, T_opt_R, T_br_R):
    return m_min_R + c_R * ((T - T_opt_R) / T_br_R)**2

def mortality_A(T, m_min_A, c_A, T_opt_A, T_br_A):
    return m_min_A + c_A * ((T - T_opt_A) / T_br_A)**2
