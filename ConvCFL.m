function [X,Gamma,C,D, res] = ConvCFL(D, C, I, lamb1, lamb2, opts)
% COUPLED FEATURE LEARNING VIA STRUCTURED CONVOLUTIONAL SPARSE CODING
%
% Problem:
%  Decomposition of multimodal images I_1, I_2, ..., I_n into their
%  correlated and independent components.
%
% Inputs:
%   I:                input images (I_i is in I(:,:,i)) (N1 x N2 x n)
%   D:                coupled dictionaries (D_i is in D(:,:,:,i), and D(:,:,k,i) is the k-th filter in D_i) (m x m x K x n)
%   C:                common dictionary (for independent features) (m x m x L)
%   lamb1:            sparsity regularization parameter for Gamma (common sparse codes)
%   lamb2:            sparsity regularization parameter for X (modality specific sparse codes)
%   (optionals:)
%   opts.MaxIter      maximum algorithm iterations (default 150)
%   opts.csc_iters    number of CSC iterations in each cycle (default 1)
%   opts.cdl_iters    number of DL iterations in each cycle (default 1)
%
% Outputs
%   Gamma             common sparse codes (N1 x N2 x K)
%   X                 modality specific sparse codes (N1 x N2 x L x n)
%   D                 coupled dictionaries
%   C                 common dictionaries
%___________________________________________________________________________

%% parameters
[N1,N2,n] = size(I);
m = size(D,1); % filter size
K = size(D,3); % number of filers
L = size(C,3); % number of filers
I = reshape(I,[N1 N2 1 n]);

if nargin < 4
    opts = [];
end
if ~isfield(opts,'MaxIter')
    opts.MaxIter = 150;
end
if ~isfield(opts,'csc_iters')
    opts.csc_iters = 1;
end
if ~isfield(opts,'cdl_iters')
    opts.cdl_iters = 1;
end
if ~isfield(opts,'rho')
    opts.rho = 10; % penalty param CSC
end
if ~isfield(opts,'sig')
    opts.sig = 10; % penalty param DL
end
if ~isfield(opts,'AutoRho')
    opts.AutoRho = 1; % varying penalty parameter CSC
end
if ~isfield(opts,'AutoSig')
    opts.AutoSig = 1; % varying penalty parameter DL
end
if ~isfield(opts,'Xinit')
    opts.Xinit = zeros(N1,N2,L,n,'single');
end
if ~isfield(opts,'Gammainit')
    opts.Gammainit = zeros(N1,N2,K,'single');
end
if ~isfield(opts,'Uinit')
    opts.Uinit = zeros(N1,N2,K,n,'single');
end
if ~isfield(opts,'Tinit')
    opts.Tinit = zeros(N1,N2,L,n,'single');
end
if ~isfield(opts,'Rinit')
    opts.Rinit = zeros(N1,N2,K,n,'single');
end
if ~isfield(opts,'Vinit')
    opts.Vinit = zeros(N1,N2,L,n,'single');
end
if ~isfield(opts,'relaxParam')
    opts.relaxParam = 1.8; % over relaxation parameter
end

%% initialization
If = fft2(I);
X = opts.Xinit; % individual sparse code
Gamma = opts.Gammainit; %common sparse codes
U = opts.Uinit; % scaled dual variable
V = opts.Vinit; % scaled dual variable
T = opts.Tinit; % scaled dual variable
R = opts.Rinit; % scaled dual variable


rho = opts.rho;
sig = opts.sig;

Maxitr = opts.MaxItr;
csc_iters = opts.csc_iters;
cdl_iters = opts.cdl_iters;

res.iterinf = [];

mu = 10; % mu and tau are used in varying penalty parameter (ADMM extension)
tau = 1.2;

alpha = opts.relaxParam; % over-relaxation parameter (ADMM extension)

vec = @(x) x(:);
itr = 1;

%% CDL CYCLES
tsrt = tic;

D = padarray(D,[N1-m N2-m],'post');
C = padarray(C,[N1-m N2-m],'post');
Df = fft2(D);
Cf = fft2(C);

while itr<=Maxitr
    %%% ADMM iterations

    %% CSC
    for ttt = 1:csc_iters % default = 1
        Xprv = X;
        Gammaprv = Gamma;
        
        Z  = Z_update(fft2(cat(3,Gamma-U,X-V)),cat(3,Df,repmat(Cf,1,1,1,n)),If,rho) ; % Z update
        Zr = alpha * Z + (1-alpha)*cat(3,repmat(Gamma,1,1,1,n),X); % relaxation
        
        Gamma = sfthrsh(mean(Zr(:,:,1:K,:)+U,4), lamb1/rho); % X update
        X = sfthrsh(Zr(:,:,K+1:end,:)+V, lamb2/rho); % X update

        U =  Zr(:,:,1:K,:)- Gamma + U; % U update
        V =  Zr(:,:,K+1:end,:) - X + V; % V update
    end
    if csc_iters == 0
        Z = cat(3,repmat(Gamma,1,1,1,n),X);
        Xprv = X;
        Gammaprv = Gamma;
    end    
    %% CDL
    for ttt = 1:cdl_iters % default = 1
        Dprv = D;
        Cprv = C;
        
        E = E_update(fft2(cat(3,repmat(Gamma,1,1,1,n),X)),fft2(cat(3,D-R,C-T)),If,sig);
        Er = alpha * E + (1-alpha)*cat(3,D,repmat(C,1,1,1,n)); % relaxation
        
        D = D_proj(Er(:,:,1:K,:)+R,m,N1,N2,1); % projection on constraint set     
        C = D_proj(mean(Er(:,:,K+1:end,:)+T,4),m,N1,N2,1); % projection on constraint set
                                
        R = Er(:,:,1:K,:) - D + R;
        T = Er(:,:,K+1:end,:) - C + T;
        
    end
    if cdl_iters == 0
        E = cat(3,D,repmat(C,1,1,1,n));
        Dprv = D;
        Cprv = C;
    end
    %%
    Df = fft2(D);
    Cf = fft2(C);
    titer = toc(tsrt);
    %%
    
    %_________________________residuals CSC_____________________________
    nX = norm(Z(:)); nZ = norm([X(:) ; Gamma(:)*(n)]); nUV = norm([U(:); V(:)]);
    r_csc = norm(vec(Z-cat(3,repmat(Gamma,1,1,1,n),X)))/(max(nX,nZ)); % primal residulal
    s_csc = norm([vec(Xprv-X);  vec(Gammaprv-Gamma)*(n)])/nUV; % dual residual
    
    %_________________________residuals CDL_____________________________
    nE = norm(E(:)); nD = norm( [D(:); C(:)*(n)] ) ; nRT = norm([R(:); T(:)]);
    r_cdl = norm(vec(E-cat(3,D,repmat(C,1,1,1,n))))/(max(nE,nD)); % primal residulal
    s_cdl = norm([vec(Cprv-C)*(n);  vec(Dprv-D)])/nRT; % dual residual
    
    %_________________________rho update_____________________________
    if opts.AutoRho && rem(itr,5)==0
        [rho,U,V] = rho_update(rho,r_csc,s_csc,mu,tau,U,V);
    end
    
    %_________________________sig update_____________________________
    if opts.AutoSig && rem(itr,5)==0
        [sig,R,T] = rho_update(sig,r_cdl,s_cdl,mu,tau,R,T);
    end
    
    %_________________________progress_______________________________
    rPow = sum(vec(abs(sum(Df.*fft2(Gamma),3) + sum(Cf.*fft2(X),3)-If).^2))/(2*N1*N2); % residual power
    L1 = lamb2*sum(abs(X(:)))+lamb1*sum(abs(Gamma(:))); % l1-norm
    fval = rPow + L1; % functional value
    res.iterinf = [res.iterinf; [itr fval rPow L1 r_csc s_csc r_cdl s_cdl rho sig titer]];
    
    
    itr = itr+1;
end
D = D(1:m,1:m,:,:);
C = C(1:m,1:m,:);
end

function y = sfthrsh(x, kappa) % shrinkage operator
y = sign(x).*max(0, abs(x) - kappa);
end

function Z  = Z_update(Wf,Ff,If,rho)
B = conj(Ff)./(sum(abs(Ff).^2,3)+rho);
Rf = If - sum(Wf.*Ff,3); % residual update
Zf = Wf + B.*Rf; % X update
Z  = ifft2(Zf,'symmetric');
end

function E = E_update(Sf,Qf,If,sig)
C = conj(Sf)./(sum(abs(Sf).^2,3)+sig);
Rf = If - sum(Sf.*Qf,3); % residual update
Gf = Qf + C.*Rf;
E = ifft2(Gf,'symmetric');
end

function D = D_proj(D,M,N1,N2,cons) % projection on unit ball
D = padarray(D(1:M,1:M,:,:),[N1-M N2-M],'post');
D  = D./max(sqrt(sum(D.^2,1:2)),cons);
end


function [rho,U,V] = rho_update(rho,r,s,mu,tau,U,V)
% varying penalty parameter (ADMM extension)
a = 1;
if r > mu*s
    a = tau;
end
if s > mu*r
    a = 1/tau;
end
rho_ = a*rho;
if rho_>1e-4
    rho = rho_;
    U = U/a;
    V = V/a;
end
end