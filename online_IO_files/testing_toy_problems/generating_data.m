%%% generating_data.m %%%
% 5/18/2019

%This file was used to generate some data for some of the 
%unit tests for the Online_IO class.

%For more information, see the Validation or
%Validation/Usage subsection of the Sections
%1.2, 1.3, and 1.4 of the Chapter documentation
%When we used it, we referred to it as "the 
%MATLAB file" or "the MATLAB script" or some 
%other variation.  Since it was used throughout
%the unit tests, there isn't a central description
%for it

%%
%%% Generating Data to Test the .reconstruct()
%%% method
% Utilizing the form found for the Nocedal and Wright problem
% See the Validation subsection in Section 1.2 for the formula
% (where I'm explaining the N&W problem)

%%% Nocedal and Wright Problem %%%
lb = -20;
ub = 20;

Q = [6 2 1; 2 5 2; 1 2 4];
c = [8; 3; 3];

% Generate some A constraint data (A matrix is constr) %
%constr = [1 0 1; 0 1 1]; %OG form
constr = [randi([lb ub]) 0 randi([lb ub]); 0 randi([lb ub]) randi([lb ub])];
disp('A_constr=')
disp(constr)

% Generating some b RHS data (b_2 = b) %
%b_2 = [3; 0]; %OG form
b_2 = [randi([lb ub]); 0];
disp('b_constr=')
disp(b_2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Calculating the Analytical Solution %%%
lhs_dual = constr*inv(Q)*constr';
rhs_dual = b_2 + constr*inv(Q)*(-1*c);

lambda = lhs_dual\rhs_dual;

disp('lambda=')
disp(lambda)

x = inv(Q)*(constr'*lambda-(-1*c));

disp('x=')
disp(x)


%%
%Moving the (4) Constraint in Unit Test Toy Problem #1 (CLT)
%We then solve a linear system to obtain the new solution
%See Validation subsection of Section 1.2 for more information
%and see the test_receive_data_CLT_DCZ and 
%test_receive_data_CLT_DCZ_non_negative unittests in that section

constr1 = [2 5];
constr2 = [2 -3];
constr3 = [2 1];
constr4 = [-2 -1]; %change this to [-1.7 -1] to get slightly diff sol
constr4_modified = [-1.8 -1]; %I think since we are really testing
%expressions, we don't need to change this RHS

rhs1 = 10;
rhs2 = -6;
rhs3 = 4;
rhs4 = -10;

%%% New A %%%
Anew = [constr2; constr4_modified];
b = [rhs2; rhs4];

x_sol = Anew\b;

disp("x_sol=")
disp(x_sol)


%%
%Data for testing the project_to_F function
%See Validation/Usage subsection of Section 1.3
%Thanks to 
%https://www.mathworks.com/help/matlab/matlab_prog/anonymous-functions.html
%for being a good reference for the @ functions

proj_F = @(x) (x(1) - (-8))^2 + (x(2) - (-3))^2 + (x(3) - (-3))^2;

lb = -20;
ub = 20;

for i = 1:3
    disp("***************")
    c = [randi([lb ub]); randi([lb ub]); randi([lb ub])];
    
    value = proj_F(c);

    disp("input:")
    disp(c)

    disp("results of proj_F")
    disp(value)

end

%%
%Creating some stuff for gradient_step
%See Validation/Usage subsection of Section 1.3
OGD = @(eta,c_t,x_t,xbar_t) (c_t - eta*(x_t-xbar_t));

lb = -50;
ub = 50;

for i = 1:2
    disp("*************************")
    
    eta = round(1/(sqrt(abs(randi([lb ub])))),4);
    disp("eta")
    disp(eta)

    c_t = [randi([lb ub]); randi([lb ub])];
    disp("c_t")
    disp(c_t)

    x_t = [randi([lb ub]); randi([lb ub])];
    disp("x_t")
    disp(x_t)

    xbar_t = [randi([lb ub]); randi([lb ub])];
    disp("xbar_t")
    disp(xbar_t)

    value = OGD(eta,c_t,x_t,xbar_t);
    disp("value")
    disp(value)
    
    disp("*************************")

end

%%
%Data for the loss_function objective function
%See Validation/Usage subsection of Section 1.4

update_func = @(x) ((2.5-x(1))^2 + (3-x(2))^2);

lb = -50;
ub = 50;

for i = 1:3
    disp("***************")
    c = [randi([lb ub]); randi([lb ub])];
    
    value = update_func(c);

    disp("input:")
    disp(c)

    disp("results of loss_func")
    disp(value)

end

%%
%Data for update_rule_optimization_model objective function
%See Validation/Usage subsection of Section 1.4


update_func = @(c,x) (0.5*((c(1)-2)^2 + (c(2)+3)^2) + ...
    (5/sqrt(30))*((2.5-x(1))^2 + (3-x(2))^2));

lb = -30;
ub = 30; %make the ranges a little smaller bc
%I don't want the objective func value to get too big

for i = 1:3
    disp("***************")
    %%% Input %%%
    c = [randi([lb ub]); randi([lb ub])];
    x = [randi([lb ub]); randi([lb ub])];
    
    disp("c:")
    disp(c)
    
    disp("x:")
    disp(x)
    
    %%% Evaluate Func %%%
    value = update_func(c,x);

    disp("results of update_func")
    disp(value)

end





















%%% 