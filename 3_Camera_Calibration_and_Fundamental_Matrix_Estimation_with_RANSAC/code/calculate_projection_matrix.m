% Projection Matrix Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu, Grady Williams, James Hays

% Returns the projection matrix for a given set of corresponding 2D and
% 3D points. 

% 'Points_2D' is nx2 matrix of 2D coordinate of points on the image
% 'Points_3D' is nx3 matrix of 3D coordinate of points in the world

% 'M' is the 3x4 projection matrix


function M = calculate_projection_matrix( Points_2D, Points_3D )

% To solve for the projection matrix. You need to set up a system of
% equations using the corresponding 2D and 3D points:

%                                                     [M11       [ u1
%                                                      M12         v1
%                                                      M13         .
%                                                      M14         .
%[ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1          M21         .
%  0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1          M22         .
%  .  .  .  . .  .  .  .    .     .      .          *  M23   =     .
%  Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn          M24         .
%  0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]        M31         .
%                                                      M32         un
%                                                      M33         vn ]

% Then you can solve this using least squares with the '\' operator or SVD.
% Notice you obtain 2 equations for each corresponding 2D and 3D point
% pair. To solve this, you need at least 6 point pairs.

%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

% Construct Matrices: Am = b
[row, col] = size(Points_2D);
data_num = row;
A = zeros(data_num * 2, 11);
b = zeros(data_num * 2, 1);

% Construct A
for i = 1 : row
    A(i * 2 - 1, 1:3) = Points_3D(i, :);
    A(i * 2, 5:7) = Points_3D(i, :);
    A(i * 2 - 1, 4) = 1;
    A(i * 2, 8) = 1;
    A(i * 2 - 1, 9:11) = - Points_3D(i, :) * Points_2D(i, 1);
    A(i * 2, 9:11) = - Points_3D(i, :) * Points_2D(i, 2);
end

% Construct b
for i = 1 : row
    b(i * 2 - 1) = Points_2D(i, 1);
    b(i * 2) = Points_2D(i, 2);
end

% Compute equation
M = A \ b;
M = [M;1];
M = reshape(M,[],3)';

end

