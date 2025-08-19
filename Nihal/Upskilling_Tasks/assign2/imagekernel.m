close all;
n = 20;
m = 40;
image = im2double(imread("image.jpg"));
K_edge = [-1  0  1; -2  0  2; -1  0  1];
K_gauss = (1/16) * ([1;2;1]*[1 2 1]);
K_average = (1/(n*n))*ones(n,n);
edge_image = zeros(size(image));
gauss_image = zeros(size(image));
average_image = zeros(size(gauss_image));
for c=1:3
    edge_image(:,:,c) = abs(conv2(image(:,:,c), K_edge, 'same'));
    temp = image(:,:,c);
    for k=1:m
        temp = conv2(temp, K_gauss, 'same');
    end
    gauss_image(:,:,c) = temp;
    average_image(:,:,c) = conv2(image(:,:,c), K_average, 'same');
end
figure;
subplot(4,1,1);
imshow(image);
title("Image");
subplot(4,1,2);
imshow(gauss_image);
title("Gaussian Blur x"+m);
subplot(4,1,3);
imshow(edge_image);
title("Edge");
subplot(4,1,4);
imshow(average_image);
title("Averaged convolve "+n+"x"+n);
figure;
subplot(3,1,1);
hold on;
histogram(image(:,:,1), 256);
histogram(edge_image(:,:,1),256);
histogram(gauss_image(:,:,1), 256);
legend('Original', 'Edge', 'Gaussian Blur');
title('Red Channel');
xlim([0 1]);
hold off;

subplot(3,1,2);
hold on;
histogram(image(:,:,2), 256);
histogram(edge_image(:,:,2),256);
histogram(gauss_image(:,:,2), 256);
legend('Original', 'Edge', 'Gaussian Blur');
title('Green Channel');
xlim([0 1]);
hold off;

subplot(3,1,3);
hold on;
histogram(image(:,:,3), 256);
histogram(edge_image(:,:,3),256);
histogram(gauss_image(:,:,3), 256);
legend('Original', 'Edge', 'Gaussian Blur');
title('Blue Channel');
xlim([0 1]);
hold off;