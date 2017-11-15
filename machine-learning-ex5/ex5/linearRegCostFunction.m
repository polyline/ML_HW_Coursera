function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X => m x n
% theta => n x 1
% h => m x 1

h = X * theta;

% theta 0 should be ignored in regularization term
% But we don't want to change the size of theta, so set the first element as 0
theta(1, :) = 0;

J = (1/(2*m)) .* sum((h-y).^2) + (lambda /(2*m)) * sum(theta.^2); 

% grad => n x 1
% Don't need sum..., after matrices multipulation, already summed up.

grad = (1/m) * X' * (h-y) + (lambda / m) .* theta;








% =========================================================================

grad = grad(:);

end
