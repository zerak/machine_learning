% algorithm 2

% COSTFUNCTION Compute cost and gradient for logistic regression
% J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the parameter for logistic regression 

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

function value = cost_function(theta, x, y)

% number of training examples
m = length(y);
value = 0;
for i = 1:m
    sigmoid_value = sigmoid(theta * x(i, :)');
    value += y(i)*log(sigmoid_value) + (1-y(i))*log(1-sigmoid_value);
end
value = -value / m;

end
