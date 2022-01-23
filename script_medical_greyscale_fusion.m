% Code for multimodal medical greyscale image fusion (CT-MRI)

clear
% clc
addpath('utilities');

%% multimodal inputs
folder_name = 'Medical_Gray';
i1 = single(  rgb2gray(imread([folder_name '\I20.png'])) )/255; %  image 1
i2 = single(  rgb2gray(imread([folder_name '\J20.png']))  )/255; %  image 2

%% base-detail layers decomposition
tic

% lowpass filtering
fltlmbd = 10;
[i1_low, i1_high] = lowpass(i1, fltlmbd);
[i2_low, i2_high] = lowpass(i2, fltlmbd);

%% details layer
I_input = [];
I_input(:,:,1) = i1_high;
I_input(:,:,2) = i2_high;
[N1,N2,n] = size(I_input);

%% parameters
lamb1 = 0.01;
lamb2 = 0.01;
opts.MaxItr = 100;
opts.csc_iters = 1;
opts.cdl_iters = 1; % use 0 for using pre-learned dictioonaries, and use 1 for learning adaptive dictionaries

%% initial dictionaries
%%% option 1 : random dictionaries ()

% M = 8; % filter size
% K = 8; % number of filters in D
% L = 10; % number of filters in C
% D0 = repmat(initdict(M,K,0),1,1,1,n);
% C0 = initdict(M,L,0);


%%% option 2: using pre-learned dictionaries

load('dicts\D0_med_grey.mat')
load('dicts\C0_med_grey.mat')

%% Learn sparse codes and dictionaries
[X,Gamma,C,D, res] = ConvCFL(D0, C0, I_input, lamb1, lamb2, opts); %peforming decomposition
[Gamma,X,nR] = sparse_orth_proj(Gamma,X,D,C,I_input,10,0.01); % projecting ...
%on the sparse support using gradient descent method (optional)

%% Fusion
IF = fuse_grey(D,C,X,Gamma,i1_low,i2_low,0.8);
toc

imwrite(uint8(IF*255),'Results\med_grey_result.png')

% D0 = D; save('dicts\D0_med_grey_new.mat','D0');
% C0 = C; save('dicts\C0_med_grey_new.mat','C0');

figure(345)
subplot 131
imshow(i1,[0 1]),xlabel('i1')
subplot 132
imshow(i2,[0 1]),xlabel('i2')
subplot 133
imshow(IF,[0 1]),xlabel('if')
% 
% figure(365);
% subplot 131
% imshow(dict2image(C),[]); title('C')
% subplot 132
% imshow(dict2image(D(:,:,:,1)),[]); title('D_1')
% subplot 133
% imshow(dict2image(D(:,:,:,2)),[]); title('D_2')