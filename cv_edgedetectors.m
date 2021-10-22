%Jon Dexter
%June 10, 2021
clear all
%roberts, %prewitts, %sobel, %canny, %log

%performance measures
%Mean Square Error(MSE) represents the cumulative squared error between the compressed(ie edge detector image) and
%the original image.  The lower the value of MSE, the lower the error.

%The Peak Signal to Noise Ratio(PSNR) block computes the peak signal-to-noise ratio, in decibels, between two images. 
%This ratio is used as a quality measurement between the original and a compressed image(ie edge detector image). 
%The higher the PSNR, the better the quality of the compressed, or reconstructed image.


myImagee=rgb2gray(imread('lena1.jpg')); %image read as grayscale
%subplot(3,3,[1,2,3]), imshow(myImagee), title('Original Image');
subplot(3,3,[1,2,3]), imshow(myImagee), title('Original Image');

myImage = double(myImagee); % convert from unit8 to double

%Roberts operator
%-------------------------------------------------------------------
filtered_image = zeros(size(myImage)); % Pre-allocate the filtered_image matrix with zeros

% Robert Operator Mask
Mx = [1 0; 0 -1];
My = [0 1; -1 0];

% Edge Detection Process
% When i = 1 and j = 1, then filtered_image pixel position will be filtered_image(1, 1)
% The mask is of 2x2, so we need to traverse % to % filtered_image(size(myImage, 1) - 1 %, size(myImage, 2) - 1)
for i = 1:size(myImage, 1) - 1
	for j = 1:size(myImage, 2) - 1

		% Gradient approximations
		Gx = sum(sum(Mx.*myImage(i:i+1, j:j+1)));
		Gy = sum(sum(My.*myImage(i:i+1, j:j+1)));
				
		% Calculate magnitude of vector
		filtered_image(i, j) = sqrt(Gx.^2 + Gy.^2);
	end
end

filtered_image = uint8(filtered_image); % Displaying Filtered Image

% Define a threshold value
thresholdValue = 100; % varies between [0 255]
output_image = max(filtered_image, thresholdValue);
output_image(output_image == round(thresholdValue)) = 0;
roberts =output_image;
%roberts = edge(myImage, 'roberts', 0.9);
[MSE , PSNR_robs] = performance(myImage, roberts);
subplot(3,3,4), imshow(roberts), title('Roberts operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_robs));
%---------------------------------------------------------------------------------------------------------------------






%Prewitt operator
%---------------------------------------------------------------------------------------------------------------------
filtered_image = zeros(size(myImage)); % Pre-allocate the filtered_image matrix with zeros
% Prewitt Operator Mask
Mx = [-1 0 1; -1 0 1; -1 0 1];
My = [-1 -1 -1; 0 0 0; 1 1 1];
% Edge Detection Process
% When i = 1 and j = 1, then filtered_image pixel  position will be filtered_image(2, 2)
% The mask is of 3x3, so we need to traverse % to filtered_image(size(myImage, 1) - 2
%, size(myImage, 2) - 2)
% Thus we are not considering the borders.
for i = 1:size(myImage, 1) - 2
	for j = 1:size(myImage, 2) - 2

		% Gradient approximations
		Gx = sum(sum(Mx.*myImage(i:i+2, j:j+2)));
		Gy = sum(sum(My.*myImage(i:i+2, j:j+2)));
				
		% Calculate magnitude of vector
		filtered_image(i+1, j+1) = sqrt(Gx.^2 + Gy.^2);
		
	end
end

% Displaying Filtered Image
filtered_image = uint8(filtered_image);

% Define a threshold value
thresholdValue = 100; % varies between [0 255]
output_image = max(filtered_image, thresholdValue);
output_image(output_image == round(thresholdValue)) = 0;
prewitt = output_image;
prewitt = edge(myImage,'prewitt'); 
[MSE , PSNR_prewit] = performance(myImage, prewitt);
subplot(3,3,5), imshow(prewitt), title('Prewitt operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_prewit)); 
%---------------------------------------------------------------------------------------------------------------------




%sobel operator
%---------------------------------------------------------------------------------------------------------------------
filtered_image = zeros(size(myImage));   % Pre-allocate the filtered_image matrix with zeros
% Sobel Operator Mask
Mx = [-1 0 1; -2 0 2; -1 0 1];
My = [-1 -2 -1; 0 0 0; 1 2 1];

% Edge Detection Process
% When i = 1 and j = 1, then filtered_image pixel  position will be filtered_image(2, 2)
% The mask is of 3x3, so we need to traverse  to % filtered_image(size(myImage, 1) - 2 , size(myImage, 2) - 2)
% Thus we are not considering the borders.
for i = 1:size(myImage, 1) - 2
	for j = 1:size(myImage, 2) - 2

		% Gradient approximations
		Gx = sum(sum(Mx.*myImage(i:i+2, j:j+2)));
		Gy = sum(sum(My.*myImage(i:i+2, j:j+2)));
				
		% Calculate magnitude of vector
		filtered_image(i+1, j+1) = sqrt(Gx.^2 + Gy.^2);
		
	end
end
filtered_image = uint8(filtered_image); % Displaying Filtered Image

% Define a threshold value
thresholdValue = 100; % varies between [0 255]
output_image = max(filtered_image, thresholdValue);
output_image(output_image == round(thresholdValue)) = 0;
sobel = output_image;
sobel = edge(myImage,'sobel');
[MSE , PSNR_sobel] = performance(myImage, sobel);
subplot(3,3,6), imshow(sobel), title('Sobel operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_sobel)); 
%---------------------------------------------------------------------------------------------------------------------

%canny operator
%-----------------------------------------------------------------------------------------------------------------------
canny = edge(myImage,'canny',[]);
[MSE , PSNR_canny] = performance(myImage, canny);
subplot(3,3,7), imshow(canny, []), title('Canny edge operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_canny));
%-----------------------------------------------------------------------------------------------------------------------

%log operator
%----------------------------------------------------------------------------------------------------------------------
log = edge(myImage,'log', []);
[MSE , PSNR_log] = performance(myImage, log);
subplot(3,3,8), imshow(log), title('LoG edge operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_log)); 
%----------------------------------------------------------------------------------------------------------------------


%DoG
%----------------------------------------------------------------------------------------------------------------------
G1= fspecial('gaussian',21,15);
G2 = fspecial('gaussian',21,20);
DoG = G1 - G2;
myImage_DoG = conv2(myImage,DoG,'same');
[MSE , PSNR_DoG] = performance(myImage, myImage_DoG);
subplot(3,3,9), imshow(myImage_DoG,[]), title('DoG edge operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_DoG));
%----------------------------------------------------------------------------------------------------------------------



function [mse, psnr] = performance(myImage, edgeImage)
%function to calculate the performace measure, ie mse and psnr
[rows, columns] = size(myImage);
squaredErrorImage = (double(myImage) - double(edgeImage)) .^ 2;

% Sum the Squared Image and divide by the number of elements
% to get the Mean Squared Error.  It will be a scalar (a single number).
mse = double(sum(sum(squaredErrorImage)) / (rows * columns));

% Calculate PSNR (Peak Signal to Noise Ratio) from the MSE according to the formula.
psnr = 10 * log10( 256^2 / mse);
fprintf('The mean square error is %f\nThe psnr = %f', mse, psnr);
end