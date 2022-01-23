function IF = fuse_grey(D,C,X,Gamma,i1_low,i2_low,w)
% w: wighting parameter
N1 = size(X,1);
N2 = size(X,2);
X1 = X(:,:,:,1); X2 = X(:,:,:,2);
D1 = D(:,:,:,1); D2 = D(:,:,:,2);

if_base_min = min(i1_low,i2_low);
if_base_max = max(i1_low,i2_low);


XF = zeros(N1,N2,size(X,3));


%% Fusion

XF(abs(X1)>= abs(X2)) = X1(abs(X1)>= abs(X2));
XF(abs(X1)< abs(X2)) = X2(abs(X1)< abs(X2));
nd1 = sqrt(sum(D1.^2,1:2));
nd2 = sqrt(sum(D2.^2,1:2));
DF = D1.*(nd1>=nd2) + D2.*(nd1<nd2);

if_details = ifft2(sum(fft2(C,N1,N2).*fft2(XF),3) + sum(fft2(DF,N1,N2).*fft2(Gamma),3),'symmetric');

[~, ~, if_det_range] = LocMinMax(if_details,1);
if_det_range = imgaussfilt(if_det_range,1); % smoothing

if_base = min(1-if_det_range,if_base_max); % highest local intensity is the max intensity


%%% weighted averaging
% w = 0   lowest local intensity is the min intensity
% w = 0.5 lowest local intensity is the average of min and max intensity
% w = 1   lowest local intensity is the max intensity
if_base = max(if_base,w*if_base_max+(1-w)*if_base_min);
if_base = imgaussfilt(if_base,.01,'FilterSize',[5 5]); % smoothing base-layer

IF = if_base + if_details;
IF(IF<0) =  0;
IF(IF>1) =  1;
end


function [Imax, Imin, Irange] = LocMinMax(I,r)
% code for finding local minimum and maximums and range
% r is the radius of neighborhood

w = 2*r+1;
I = padarray(I,[r r],'replicate','both');

[H, W] = size(I);
Imax = zeros(H-2*r,W-2*r);
Imin = zeros(H-2*r,W-2*r);


for i = 1:H-w+1
    for j = 1:W-w+1
    
        b = I(i:i+w-1,j:j+w-1);
        Imax(i,j) = max(b(:));
        Imin(i,j) = min(b(:));
    end
end

Irange = Imax-Imin;

end