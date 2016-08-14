% COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
% J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
% parameter for linear regression to fit the data points in X and y

% Initialize some useful values
% You need to return the following variables correctly 
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = 0;
for i = 1:m
    J = J + (theta'*X(i, :)' - y(i))^2;
end
J = J / (2*m);

end
