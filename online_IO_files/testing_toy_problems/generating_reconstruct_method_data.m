%%% Generating Data to Test the .reconstruct()
%%% Method upon

%%% Nocedal and Wright Problem %%%
lb = -20;
ub = 20;

Q = [6 2 1; 2 5 2; 1 2 4];
%constr = [1 0 1; 0 1 1]; %OG form
constr = [randi([lb ub]) 0 randi([lb ub]); 0 randi([lb ub]) randi([lb ub])];
disp('A_constr=')
disp(constr)

A = [Q; constr];
%maybe get some random integers in those
%1 slots

c = [8; 3; 3];
%b_2 = [3; 0]; %OG form
b_2 = [randi([lb ub]); 0];
disp('b_constr=')
disp(b_2)

b = [c; b_2];
%maybe change this up a bit too

%pinv_sol = pinv(A)*b; %nope not doing what we were hoping

%disp('sol=')
%disp(x)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Trying out the analytical form I found
lhs_dual = constr*inv(Q)*constr';
rhs_dual = b_2 + constr*inv(Q)*(-1*c);

lambda = lhs_dual\rhs_dual;

disp('lambda=')
disp(lambda)

x = inv(Q)*(constr'*lambda-(-1*c));

disp('x=')
disp(x)

%%
Q = [6 2 1; 2 5 2; 1 2 4];
c = [-8 -3 -3]';

disp("**************************")
test = [6 2 1; 2 5 2; 1 2 4];
[V,D] = eig(test);

x = [-25 -25 -25]';
disp('x=')
disp(x)

value = -1*x'*V*D*V'*x;

disp("Value = ")
disp(value)

%%
%Moving the (4) Constraint in Unit Test Toy Problem #1
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
%Data for the project_to_F function

proj_F = @(x) (x(1) - (-8))^2 + (x(2) - (-3))^2 + (x(3) - (-3))^2;

lb = -20;
ub = 20;

for i = 1:3
    disp("***************")
    input = [randi([lb ub]); randi([lb ub]); randi([lb ub])];
    
    value = proj_F(input);

    disp("input:")
    disp(input)

    disp("results of proj_F")
    disp(value)

end

%%
%Creating some stuff for gradient_step
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








%%% 