% algorithm 3

% COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
% J = COSTFUNCTIONREG(theta, x, y, lambda) computes the cost of using
% theta as the parameter for regularized logistic regression and the
% gradient of the cost w.r.t. to the parameters. 

% a. theta is 1*(n+1) vector 
%    [theta0, theta1, ... thetan]
% b. x is m*(n+1) matrix
%    set x0 = 1 
%    [1, x1, ... xn]
%    [1, x1, ... xn]
%         ......
%    [1, x1, ... xn]
%    [1, x1, ... xn]
% c. y is 1*m vector 
%    [y1, y2, ... ym]

% attention: octave require function name equal to file name

% return real number

function value = cost_function_reg(theta, x, y, lambda)

% number of training examples
m = length(y);
n = length(theta);

value = 0;
for i = 1:m
    sigmoid_value = sigmoid(theta' * x(i, :)');
    value += y(i)*log(sigmoid_value) + (1-y(i))*log(1-sigmoid_value);
end
value = -value / m;

punish_value = 0;
for j = 2:n
    punish_value += theta(j)^2;
end

value = value + punish_value*lambda/(2*m);

end
