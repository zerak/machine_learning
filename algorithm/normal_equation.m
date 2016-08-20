% algorithm 4

% Normal equation compute theta 
% theta = feature_normalize(x) 

% a. x is m*(n+1) matrix
%    set x0 = 1
%    [1, x1, ... xn]
%    [1, x1, ... xn]
%         ......
%    [1, x1, ... xn]
%    [1, x1, ... xn]

% b. y is 1*m vector 
%    [y1, y2, ... ym]

% attention:
% 1). 我们使用pinv求解矩阵的逆, 当矩阵为奇异矩阵的时候求出的逆矩阵维'伪逆'
% 2). pinv和inv求出的结果有差异

% return theta 1*(n+1) vector

function theta = normal_equation(x, y)

pinv(x'*x)
theta = (pinv(x'*x) * (x' * y'))';

end
