% Gradient descent to learn theta
% theta = gradient_descent(x, y, theta, alpha)
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
% d. alpha is real number


function theta = gradient_descent(x, y, theta, alpha)

% number of training examples
m = length(y);
tmp_theta = theta;
for j = 1:length(theta)
    sum_value = 0.0;
    for i = 1:m
        sum_value = sum_value + ((tmp_theta*x(i,:)')-y(i)) * x(i, j);
    end
    % synchronous update theta
    theta(j) = theta(j) - alpha*sum_value/m;
end

end
