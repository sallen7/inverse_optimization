%%%Validation of the calculate_rho_r method for
%%%the GIO class

x0_1 = [3,2]';
x0_2 = [4,1]';
x0_3 = [2,2]';

rho_r_1 = 1 - (abs([-2,-1]*x0_1*(1/-10) - 1)/...
    calculating_denom(x0_1));
%%%rho_r_1 = 0.7143

rho_r_2 = 1 - (abs([-2,-1]*x0_2*(1/-10) - 1)/...
    calculating_denom(x0_2));
%%%rho_r_2 = 0.8852

rho_r_3 = 1 - (abs([2,5]*x0_3*(1/10) - 1)/...
    calculating_denom(x0_3));
%%%rho_r_3 = 0.1864


function denom = calculating_denom(x0)

denom = abs([2,5]*x0*(1/10) - 1) +...
    abs( [2,-3]*x0*(1/-6) - 1) +...
    abs( [2,1]*x0*(1/4) - 1) +...
    abs( [-2,-1]*x0*(1/-10) - 1);

denom = denom*(1/4);

end

