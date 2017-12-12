% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale. For best performance, you should sample random negative
% examples at multiple scales.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
% 'non_face_scn_path' is a string. This directory contains many images
%   which have no faces in them.
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.
% 'num_samples' is the number of random negatives to be mined, it's not
%   important for the function to find exactly 'num_samples' non-face
%   features, e.g. you might try to sample some number from each image, but
%   some images might be too small to find enough.

% 'features_neg' is N by D matrix where N is the number of non-faces and D
% is the template dimensionality, which would be
%   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
% if you're using the default vl_hog parameters

% Useful functions:
% vl_hog, HOG = VL_HOG(IM, CELLSIZE)
%  http://www.vlfeat.org/matlab/vl_hog.html  (API)
%  http://www.vlfeat.org/overview/hog.html   (Tutorial)
% rgb2gray

% Define variables
zoom_scale = 0.9; % Zoom out when < 0
feature_count = 1;

image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
num_images = length(image_files);
D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
feature_per_img = num_samples / num_images;
features_neg = zeros(num_images, D);

for i = 1 : num_images
    img = im2single(rgb2gray(imread(fullfile(non_face_scn_path, image_files(i).name))));
    [row, col] = size(img);
    % If the picture is not big enough, use as many features as possible.
    max_features_num = floor(row / feature_params.template_size) * floor(col / feature_params.template_size);
    expect_features_num = feature_per_img / 2;
    features_to_be_picked = min(max_features_num, expect_features_num);
    % Loop when scaling is possible
    while row >= feature_params.template_size && col >= feature_params.template_size
        if features_to_be_picked > 0
            for j = 1 : features_to_be_picked
                % Randomly choose a start feature.
                x = ceil(rand() * (col - feature_params.template_size)) + 1;
                y = ceil(rand() * (row - feature_params.template_size)) + 1;
                curr_feature = vl_hog(img(y : y + feature_params.template_size - 1, ...
                    x : x + feature_params.template_size - 1), ...
                    feature_params.hog_cell_size);
                % Add feature to the result set.
                features_neg(feature_count, :) = reshape(curr_feature, D, 1, 1);
                feature_count = feature_count + 1;
            end
            % Preparing for the next loop.
            img = imresize(img, zoom_scale);
            [row, col] = size(img);
            max_features_num = floor(row / feature_params.template_size) * floor(col / feature_params.template_size);
            expect_features_num = features_to_be_picked * zoom_scale * zoom_scale;
            features_to_be_picked = min(max_features_num, expect_features_num);
        end
    end
end
% If we didn't collect num_samples samples, use as many as we have.
N_idx = randi(size(features_neg, 1), [1, min(feature_count, num_samples)]);
features_neg = features_neg(N_idx, :);
end



