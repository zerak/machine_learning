% algorithm 3

% Normalizes the features in x 
% x = feature_normalize(x) 
% a. x is m*n matrix
%    [x1, ... xn]
%    [x1, ... xn]
%         ......
%    [x1, ... xn]
%    [x1, ... xn]

% attention: A .- B 表示矩阵A和B每个元素两两相减, 要求A和B size一样
%            A ./ B 表示矩阵A和B每个元素两两相除, 要求A和B size一样

% return x m*n matrix

function x = feature_normalize(x)

mean_x = mean(x);
std_x = std(x);
m = length(x(:, 1));
for i = 1:m
     x(i, :) = ((x(i, :) .- mean_x)) ./ std_x;
end

end
