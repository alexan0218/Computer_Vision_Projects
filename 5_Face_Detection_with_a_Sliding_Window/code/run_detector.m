% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confidences, image_ids] = .... 
    run_detector(test_scn_path, w, b, feature_params)
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cell_size (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cell_size.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i. 
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.

test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

%initialize these as empty and incrementally expand them.
scaling_factor = 0.9;
confidence_threshold = 0.5;
bboxes = zeros(0,4);
confidences = zeros(0,1);
image_ids = cell(0,1);
hog_dimension = feature_params.template_size/feature_params.hog_cell_size;
D = hog_dimension ^ 2 * 31;

for i = 1:length(test_scenes)
      
    fprintf('Detecting faces in %s\n', test_scenes(i).name)
    img = imread( fullfile( test_scn_path, test_scenes(i).name ));
    img = single(img)/255;
    if(size(img,3) > 1)
        img = rgb2gray(img);
    end
    [height, width] = size(img);
    cur_bboxes = zeros(0, 4);
    cur_confidences = zeros(0, 1);
    cur_image_ids = cell(0, 1);
    % cur_scale is used to restore the orginal coordinates(before scaling).
    cur_scale = 1;
    [row, col] = size(img);
    % Loop over this picture until it is smaller than a feature.
    while(row > feature_params.template_size && col > feature_params.template_size)
        hog = vl_hog(img, feature_params.hog_cell_size);
        % Loop over hog features in current picture.
        for y = 1 : (size(hog, 1) - hog_dimension)
            for x = 1 : (size(hog, 2) - hog_dimension)
                conf = reshape(hog(y : (y + hog_dimension - 1), x : (x + hog_dimension - 1), :), 1, D, 1) * w + b;
                % Record if we are confident enough
                if (conf > confidence_threshold)
                    cur_confidences = [cur_confidences; conf];
                    cur_image_ids = [cur_image_ids; test_scenes(i).name];
                    x_min = feature_params.hog_cell_size * (x - 1);
                    y_min = feature_params.hog_cell_size * (y - 1);
                    x_max = x_min + feature_params.template_size;
                    y_max = y_min + feature_params.template_size;
                    bbox = round([x_min, y_min, x_max, y_max] * cur_scale);
                    cur_bboxes = [cur_bboxes; bbox];
                end
            end
        end
        % Scale the picture, prepare for the next loop.
        img = imresize(img, scaling_factor);
        cur_scale = cur_scale / scaling_factor;
        [row, col] = size(img);
    end
    
    %non_max_supr_bbox can actually get somewhat slow with thousands of
    %initial detections. You could pre-filter the detections by confidence,
    %e.g. a detection with confidence -1.1 will probably never be
    %meaningful. You probably _don't_ want to threshold at 0.0, though. You
    %can get higher recall with a lower threshold. You don't need to modify
    %anything in non_max_supr_bbox, but you can.
    [is_maximum] = non_max_supr_bbox(cur_bboxes, cur_confidences, [height, width]);

    cur_confidences = cur_confidences(is_maximum,:);
    cur_bboxes      = cur_bboxes(     is_maximum,:);
    cur_image_ids   = cur_image_ids(  is_maximum,:);
 
    bboxes      = [bboxes;      cur_bboxes];
    confidences = [confidences; cur_confidences];
    image_ids   = [image_ids;   cur_image_ids];
end




