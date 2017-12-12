% Local Feature Stencil Code
% CS 4476 / 6476: Computer Vision, Georgia Tech
% Written by James Hays

% Returns a set of interest points for the input image

% 'image' can be grayscale or color, your choice.
% 'feature_width', in pixels, is the local feature width. It might be
%   useful in this function in order to (a) suppress boundary interest
%   points (where a feature wouldn't fit entirely in the image, anyway)
%   or (b) scale the image filters being used. Or you can ignore it.

% 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
% 'confidence' is an nx1 vector indicating the strength of the interest
%   point. You might use this later or not.
% 'scale' and 'orientation' are nx1 vectors indicating the scale and
%   orientation of each interest point. These are OPTIONAL. By default you
%   do not need to make scale and orientation invariant local features.
function [x, y, confidence, scale, orientation] = get_interest_points(image, feature_width)

% Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
% You can create additional interest point detector functions (e.g. MSER)
% for extra credit.

% If you're finding spurious interest point detections near the boundaries,
% it is safe to simply suppress the gradients / corners near the edges of
% the image.

% The lecture slides and textbook are a bit vague on how to do the
% non-maximum suppression once you've thresholded the cornerness score.
% You are free to experiment. Here are some helpful functions:
%  BWLABEL and the newer BWCONNCOMP will find connected components in 
% thresholded binary image. You could, for instance, take the maximum value
% within each component.
%  COLFILT can be used to run a max() operator on each sliding window. You
% could use this to ensure that every interest point is at a local maximum
% of cornerness.

% 1. Image derivatives
% Get Ix and Iy
gaussian1 = fspecial('gaussian', 3.^2, 1);
gaussian2 = fspecial('gaussian', feature_width .^2, 2);
[gaussian1_x, gaussian1_y] = gradient(gaussian1);

Ix = imfilter(image, gaussian1_x);
Iy = imfilter(image, gaussian1_y);

% 2. Square of derivatives
% Get Ix^2, Iy^2 and IxIy.
Ix2 = Ix .^2;
Iy2 = Iy .^2;
IxIy = Ix .* Iy;

% 3. Gaussian filter
gIx2 = imfilter(Ix2, gaussian2);
gIy2 = imfilter(Iy2, gaussian2);
gIxIy = imfilter(IxIy, gaussian2);

% 4. Corenerness function
alpha = 0.04;
img_size = size(image);
har = gIx2 .* gIy2 - gIxIy .^ 2 - alpha .* (gIx2 + gIy2) .^ 2;

border = zeros(img_size);
border(feature_width + 1 : end - feature_width, feature_width + 1 : end - feature_width) = 1;
har = har .* border;

% 5. Non-maxima suppression
% Pick local maxima of har.
threshold = mean2(har);
har_thresholded = har > threshold;

blob = bwconncomp(har_thresholded);
blob_num = blob.NumObjects;
x = zeros(blob_num, 1);
y = zeros(blob_num, 1);
confidence = zeros(blob_num ,1);
for i = 1 : blob_num
    pixel = blob.PixelIdxList{i};
    curr_blob = har(pixel);
    [curr_confidence, position] = max(curr_blob);
    
    confidence(i) = curr_confidence;
    [y(i), x(i)] = ind2sub(img_size, pixel(position));
end

end
