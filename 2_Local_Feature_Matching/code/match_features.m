% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% 'features1' and 'features2' are the n x feature dimensionality features
%   from the two images.
% If you want to include geometric verification in this stage, you can add
% the x and y locations of the interest points as additional features.
%
% 'matches' is a k x 2 matrix, where k is the number of matches. The first
%   column is an index in features1, the second column is an index
%   in features2. 
% 'Confidences' is a k x 1 matrix with a real valued confidence for every
%   match.
% 'matches' and 'confidences' can empty, e.g. 0x2 and 0x1.
function [matches, confidences] = match_features(features1, features2)

% This function does not need to be symmetric (e.g. it can produce
% different numbers of matches depending on the order of the arguments).

% To start with, simply implement the "ratio test", equation 4.18 in
% section 4.1.3 of Szeliski. For extra credit you can implement various
% forms of spatial verification of matches.

% Set up variables
threshold = 0.6;
match_counter = 0;
matches = [];
confidences = [];
num1_keypoints = size(features1, 1);
num2_keypoints = size(features2, 1);

% Loop over every feature point from feature1
for i = 1 : num1_keypoints
    distance_matrix = features2;
    for j = 1 : num2_keypoints
        % Compute all distances
        distance_matrix(j, :) = ...
            (distance_matrix(j, :) - features1(i, :)) .^2;
    end
    sum_vector = sum(distance_matrix, 2);
    % Rank the distance to find the first two
    [sorted_distance, feature2_matches] = sort(sum_vector);
    ratio = sorted_distance(2) / sorted_distance(1);
    if 1 / ratio < threshold
        match_counter = match_counter + 1;
        matches(match_counter, :) = [i, feature2_matches(1)];
        confidences(match_counter) = ratio;
    end
end

% Sort the matches so that the most confident onces are at the top of the
% list. You should probably not delete this, so that the evaluation
% functions can be run on the top matches easily.
[confidences, ind] = sort(confidences, 'descend');
matches = matches(ind,:);