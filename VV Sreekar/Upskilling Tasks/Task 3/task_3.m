img = imread("vfcfyfyehhi31.png");
figure;
imshow(img);
title("orginal image");

img = double(img); 

kernel = [1 2 1; 2 4 2; 1 2 1];
kernel = kernel / 16;

result = zeros(size(img));

for k = 1:3
    result(:,:,k) = conv2(img(:,:,k), kernel, 'same');
end
result = uint8(result);
figure;
imshow(result);
title("convoluted image");



