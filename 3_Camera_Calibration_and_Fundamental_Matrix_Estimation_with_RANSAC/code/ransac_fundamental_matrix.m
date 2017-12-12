% RANSAC Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by Henry Hu

% Find the best fundamental matrix using RANSAC on potentially matching
% points

% 'matches_a' and 'matches_b' are the Nx2 coordinates of the possibly
% matching points from pic_a and pic_b. Each row is a correspondence (e.g.
% row 42 of matches_a is a point that corresponds to row 42 of matches_b.

% 'Best_Fmatrix' is the 3x3 fundamental matrix
% 'inliers_a' and 'inliers_b' are the Mx2 corresponding points (some subset
% of 'matches_a' and 'matches_b') that are inliers with respect to
% Best_Fmatrix.

% For this section, use RANSAC to find the best fundamental matrix by
% randomly sample interest points. You would reuse
% estimate_fundamental_matrix() from part 2 of this assignment.

% If you are trying to produce an uncluttered visualization of epipolar
% lines, you may want to return no more than 30 points for either left or
% right images.

function [ Best_Fmatrix, inliers_a, inliers_b] = ransac_fundamental_matrix(matches_a, matches_b)


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%

[row, col] = size(matches_a);
sample_num = 8;

% Set RANSAC parameters
p = 0.99;
e = 0.5;
s = 9;
% Compute iteration times, set threshold
iteration = log(1 - p) / log(1-(1 - e) ^ s);
threshold = 0.05;

Best_Fmatrix = zeros(3, 3);
highest_score = 0;

for i = 1 : iteration
    % Pick correspondence samples
    chosen_a = zeros(sample_num, 2);
    chosen_b = zeros(sample_num, 2);
    chosen_correspondences = randsample(1 : row, sample_num);
    for j = 1 : sample_num
        chosen_a(j, :) = matches_a(chosen_correspondences(j), :);
        chosen_b(j, :) = matches_b(chosen_correspondences(j), :);
    end
    % Estimate the fundamental matrix with samples
    F = estimate_fundamental_matrix(chosen_a, chosen_b);
    score = 0;
    for j = 1 : row
        % Apply the property, score the current estimation
        x1 = [matches_a(j, :), 1];
        x2 = [matches_b(j, :), 1];
        dist = abs(x2 * F * x1');
        if dist < threshold
            score = score + 1;
        end
    end
    % Update the best estimation so far
    if score > highest_score
        highest_score = score;
        Best_Fmatrix = F;
    end
end
% Extract inliers based on the best estimation
dist = zeros(row, 1);
for i = 1 : row
    x1 = [matches_a(i, :), 1];
    x2 = [matches_b(i, :), 1];
    dist(i) = abs(x2 * Best_Fmatrix * x1');
end
[sorted_list, index_list] = sort(dist);
inlier_list = find(sorted_list < threshold);
inlier_num = length(inlier_list);
% Pick the best 30 pairs
if  inlier_num > 30
    inliers_a = zeros(30, 2);
    inliers_b = zeros(30, 2);
    for k = 1 : 30
        inliers_a(k, :) = matches_a(index_list(k), :);
        inliers_b(k, :) = matches_b(index_list(k), :);
    end
else
    inliers_a = zeros(inlier_num, 2);
    inliers_b = zeros(inlier_num, 2);
    for k = 1 : inlier_num
        inliers_a(k, :) = matches_a(index_list(k), :);
        inliers_b(k, :) = matches_b(index_list(k), :);
    end
end

end
