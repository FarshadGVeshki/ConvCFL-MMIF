function [G,X,nR] = sparse_orth_proj(G,X,D,C,I,N,a)
[H,W,K] = size(I);
I = reshape(I,[H W 1 K]);
Sux = X~=0;
Sug = G~=0;
If = fft2(I);
Df = fft2(D,H,W);
Cf = fft2(C,H,W);

for i = 1:N

    Xf = fft2(X);
    Gf = fft2(G);
    
    Rf = sum(Df.*Gf,3)+ sum(Cf.*Xf,3) - If;
    
    
    dXf = conj(Cf).*Rf;
    dX = ifft2(dXf,'symmetric').*Sux;    
    X = X - a*dX;
    
    dGf = sum(conj(Df).*Rf,4);
    dG = ifft2(dGf,'symmetric').*Sug;   
    G = G - a*dG;
    
    nR(i,1) = 0.5*norm(vec(ifft2(Rf,'symmetric')))^2;


end
end
function x = vec(y)
x = y(:);
end
