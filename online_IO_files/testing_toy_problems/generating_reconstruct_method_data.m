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











%%% 