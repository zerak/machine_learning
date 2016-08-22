% algorithm 1

% Compute cost for linear regression
% J = cost_function(x, y, theta) 
% a. x is m*(n+1) matrix
%    set x0 = 1 
%    [1, x1, ... xn]
%    [1, x1, ... xn]
%         ......
%    [1, x1, ... xn]
%    [1, x1, ... xn]
% b. y is 1*m vector 
%    [y1, y2, ... ym]
% c. theta is 1*(n+1) vector 
%    [theta0, theta1, ... thetan]

% attention: octave require function name equal to file name

% return real number

function value = cost_function(x, y, theta)

m = length(y);
value = 0;
for i = 1:m
    value = value + (theta*x(i, :)'-y(i))^2;
end
value = value/(2*m);

end
