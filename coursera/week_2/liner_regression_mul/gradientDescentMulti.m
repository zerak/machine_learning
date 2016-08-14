% GRADIENTDESCENTMULTI Performs gradient descent to learn theta
% theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
% taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
% Instructions: Perform a single gradient step on the parameter vector
%               theta. 
%
% Hint: While debugging, it can be useful to print out the values
%       of the cost function (computeCostMulti) and gradient here.
%

function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    tmp_theta = theta;
    for j = 1:length(theta)
        sum_value = 0.0;
        for i = 1:m
            sum_value = sum_value + ((tmp_theta'*X(i,:)') - y(i)) * X(i, j);
        end
        theta(j) = theta(j) - alpha*sum_value/m;
    end
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
