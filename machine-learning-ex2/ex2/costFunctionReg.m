function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = zeros(size(theta' * X'));
h = sigmoid( theta' * X');
sum = 0 ;
sum1 = zeros(size(theta));
for i = 1 : m
    sum = sum + (-y(i) * log(h(i)) - (1 - y(i)) * log(1 - h(i)));
    for j = 1 : size(theta)
    sum1(j) = sum1(j) + ( h(i) - y(i)) * X(i,j);
    end
end
J = 1 / m * sum + lambda /(2 * m)* ((theta' * theta) - theta(1)*theta(1)) ;
grad = 1 / m * sum1 + lambda / m * theta;
grad(1) = 1 / m * sum1(1);





% =============================================================

end
