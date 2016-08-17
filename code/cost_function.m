% Compute cost for linear regression
% J = compute_cost(x, y, theta) 
% a. x is (n+1)*m matrix
%    [x0, x1, ... xn]
%    [x0, x1, ... xn]
%         ......
%    [x0, x1, ... xn]
%    [x0, x1, ... xn]
% b. y is [y0, y1, ... yn] vector
% c. theta is [theta0, theta1, ... thetan] vector 

function value = compute_cost(x, y, theta)

% number of training examples
m = length(y);
value = 0;
for i = 1:m
    value = value + (theta*x(i, :)' - y(i))^2;
end
value = value/(2*m);

end
