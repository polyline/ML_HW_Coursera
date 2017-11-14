function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%%%% Part 1 %%%%%

%%% Second layer %%%
% a1 = X

% X = the number of samples * the number of attr (Layer 1)
% Theta1 = the number of attr (Layer 2) * the number of attr (Layer 1)
% z2 should be = the number of samples * the number of attr (Layer 2)

a1 = [ones(size(X, 1), 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

%%% Third Layer %%%
% a3 should be a m * 10 matirx

z2 = [ones(size(z2, 1), 1) z2];
a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

%%% Compute Cost Function %%%
% a3 => m x num_labels
% y => m x 1
% so we need to create new y = m x num_labels

% It was a little bit redundant, O(m). If there is another faster way?
ny = zeros(m, num_labels);
for i = 1:m
	ny(i, y(i,1)) = 1;
end

% Actually, The cost was calculated by y (which composed by 1 & 0) - log(a3)
% If log(a3) is close to 0, the value would become a huge negative number.
% If close to 1, the value would be a tiny number.
% The front part calculates the cost of false positive
% while the back part calculates the cost of false negative 

J = (1/m) * sum(sum((-1 * ny .* log(a3)) - (1-ny) .* log(1-a3)));

%%%%% Part 2 %%%%%

% Calculate delta
delta_3 = a3 - ny;
delta_2 = delta_3 * Theta2 .* sigmoidGradient(z2);

% Remove delta_2 0
delta_2 = delta_2(:, 2:end);

% Calculate Big Delta
De_2 = delta_3' * a2 ;
De_1 = delta_2' * a1 ;


%%% Debug %%%
%fprintf('Size of Theta1: %d x %d\n',size(Theta1, 1), size(Theta1, 2));
%fprintf('Size of Theta2: %d x %d\n',size(Theta2, 1), size(Theta2, 2));
%fprintf('Size of Theta1_grad: %d x %d\n',size(Theta1_grad, 1), size(Theta1_grad, 2));
%fprintf('Size of Theta2_grad: %d x %d\n',size(Theta2_grad, 1), size(Theta2_grad, 2));

%%%%% Part 3 %%%%%

%%% Regularation of Cost Function %%%

% Should not regularize theta 0

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

J = J + (lambda / (2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));

%%% Regularation of Gradient %%%

% Calculate D / Gradient
Theta2_grad = (1/m) * De_2 + (lambda / m) * Theta2;
Theta1_grad = (1/m) * De_1 + (lambda / m) * Theta1;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
