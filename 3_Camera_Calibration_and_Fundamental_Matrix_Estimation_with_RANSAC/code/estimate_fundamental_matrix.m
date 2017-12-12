% Fundamental Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Returns the camera center matrix for a given projection matrix

% 'Points_a' is nx2 matrix of 2D coordinate of points on Image A
% 'Points_b' is nx2 matrix of 2D coordinate of points on Image B
% 'F_matrix' is 3x3 fundamental matrix

% Try to implement this function as efficiently as possible. It will be
% called repeatly for part III of the project

function [ F_matrix ] = estimate_fundamental_matrix(Points_a,Points_b)

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

[row, col] = size(Points_a);
% Compute Matrices
A = zeros(row, 9);
for i = 1 : row
    col1 = Points_a(i, 1) * Points_b(i, 1);
    col2 = Points_a(i, 2) * Points_b(i, 1);
    col3 = Points_b(i, 1);
    col4 = Points_a(i, 1) * Points_b(i, 2);
    col5 = Points_a(i, 2) * Points_b(i, 2);
    col6 = Points_b(i, 2);
    col7 = Points_a(i, 1);
    col8 = Points_a(i, 2);
    col9 = 1;
    A(i,:) = [col1, col2, col3, col4, col5, col6, col7, col8, col9];
end

[U, S, V] = svd(A);
f = V(:, end);
F = reshape(f, [3,3])';
[U, S, V] = svd(F);
S(3,3) = 0;
F_matrix = U * S * V';
        
end

