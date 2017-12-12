function output = my_imfilter(image, filter)
% This function is intended to behave like the built in function imfilter()
% See 'help imfilter' or 'help conv2'. While terms like "filtering" and
% "convolution" might be used interchangeably, and they are indeed nearly
% the same thing, there is a difference:
% from 'help filter2'
%    2-D correlation is related to 2-D convolution by a 180 degree rotation
%    of the filter matrix.

% Your function should work for color images. Simply filter each color
% channel independently.

% Your function should work for filters of any width and height
% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
% restriction makes it unambigious which pixel in the filter is the center
% pixel.

% Boundary handling can be tricky. The filter can't be centered on pixels
% at the image boundary without parts of the filter being out of bounds. If
% you look at 'help conv2' and 'help imfilter' you see that they have
% several options to deal with boundaries. You should simply recreate the
% default behavior of imfilter -- pad the input image with zeros, and
% return a filtered image which matches the input resolution. A better
% approach is to mirror the image content over the boundaries for padding.

% % Uncomment if you want to simply call imfilter so you can see the desired
% % behavior. When you write your actual solution, you can't use imfilter,
% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
% % computation. It might be slow.
%output = imfilter(image, filter);


%%%%%%%%%%%%%%%%
% Your code here
%%%%%%%%%%%%%%%%
filter_size = size(filter);
image_size = size(image);

vertical_border = floor(filter_size(1)/2);
horizontal_border = floor(filter_size(2)/2);

pad_image = padarray(image, [vertical_border, horizontal_border],'symmetric');

red = image(1);
green = image(2);
blue = image(3);

pad_red = pad_image(:,:,1);
pad_green = pad_image(:,:,2);
pad_blue = pad_image(:,:,3);

for i = 1 : image_size(1)
    for j = 1 : image_size(2)
        center_red = 0;
        center_green = 0;
        center_blue = 0;
        
        for row = 1: filter_size(1)
            for col = 1 : filter_size(2)
                center_red = center_red + filter(row, col) * pad_red(row + i - 1, col + j - 1);
                center_green = center_green + filter(row, col) * pad_green(row + i - 1, col + j - 1);
                center_blue = center_blue + filter(row, col) * pad_blue(row + i - 1, col + j - 1);
            end
        end      
        red(i,j) = center_red;
        green(i,j) = center_green;
        blue(i,j) = center_blue;
    end
end

output = cat(3, red, green, blue);
end
