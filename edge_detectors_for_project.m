%Jon Dexter
%June 10, 2021
clear all
%roberts, %prewitts, %sobel, %canny, %log
f =figure();
f.WindowState = 'maximized';
myImagee=rgb2gray(imread('dog2.jpg')); %image read as grayscale
%subplot(3,3,[1,2,3]), imshow(myImagee), title('Original Image');
subplot(2,3,1), imshow(myImagee), title('Original Image');

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
%roberts =output_image;
tic
roberts = edge(myImage, 'roberts', []);
toc
time = toc;
fprintf("\n");
[MSE , PSNR_robs] = performance(myImage, roberts);
subplot(2,3,2), imshow(roberts), title('Roberts operator'), xlabel(sprintf('MSE: %02d\n PSNR: %f\nEt: %f secs', MSE , PSNR_robs, time));
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
%prewitt = output_image;
tic
prewitt = edge(myImage,'prewitt', []); 
toc
time = toc;
fprintf("\n");
[MSE , PSNR_prewit] = performance(myImage, prewitt);
subplot(2,3,3), imshow(prewitt), title('Prewitt operator'), xlabel(sprintf('MSE: %02d\n PSNR: %f\nEt: %f secs', MSE , PSNR_prewit, time)); 
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
%sobel = output_image;
tic
sobel = edge(myImage,'sobel', []);
toc
time = toc;
fprintf("\n");
[MSE , PSNR_sobel] = performance(myImage, sobel);
subplot(2,3,4), imshow(sobel), title('Sobel operator'), xlabel(sprintf('MSE: %02d\n PSNR: %f\nEt: %f secs', MSE , PSNR_sobel, time)); 
%---------------------------------------------------------------------------------------------------------------------

%canny operator
%-----------------------------------------------------------------------------------------------------------------------
tic
canny = edge(myImage,'canny',[]);
toc
time = toc;
fprintf("\n");
[MSE , PSNR_canny] = performance(myImage, canny);
subplot(2,3,5), imshow(canny, []), title('Canny edge operator'), xlabel(sprintf('MSE: %02d\n PSNR: %f\nEt: %f secs', MSE , PSNR_canny, time));
%-----------------------------------------------------------------------------------------------------------------------

%log operator
%----------------------------------------------------------------------------------------------------------------------
tic
log = edge(myImage,'log', []);
toc
time = toc;
fprintf("\n");
[MSE , PSNR_log] = performance(myImage, log);
subplot(2,3,6), imshow(log), title('LoG edge operator'), xlabel(sprintf('MSE: %02d\n PSNR: %f\nEt: %f secs', MSE , PSNR_log, time)); 
%----------------------------------------------------------------------------------------------------------------------


%DoG
%----------------------------------------------------------------------------------------------------------------------
G1= fspecial('gaussian',21,15);
G2 = fspecial('gaussian',21,20);
DoG = G1 - G2;
myImage_DoG = conv2(myImage,DoG,'same');
[MSE , PSNR_DoG] = performance(myImage, myImage_DoG);
%subplot(3,3,9), imshow(myImage_DoG,[]), title('DoG edge operator'), xlabel(sprintf('MSE: %02d and PSNR: %f', MSE , PSNR_DoG));
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
%fprintf('The mean square error is %f\nThe psnr = %f', mse, psnr);
end