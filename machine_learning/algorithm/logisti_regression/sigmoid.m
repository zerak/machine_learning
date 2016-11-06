% algorithm 1

% SIGMOID Compute sigmoid functoon
% Instructions: Compute the sigmoid of each value of z (z can be a real number, matrix, vector or scalar).
% g(z) = 1/(1+e^-z)

% attention: octave require function name equal to file name

% return new matrix or vector

function g = sigmoid(z)

g = 1 ./ (1 .+ (e .^ -z));

end
