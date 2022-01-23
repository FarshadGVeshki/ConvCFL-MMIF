function D = initdict(M,K,dcatom)
s = max(M);
D =[];

for i = 1:length(M)
d = randn(M(i),M(i),K(i));
d  = (d)./sqrt(sum(d.^2,1:2));
d = padarray(d,[s-M(i)  s-M(i)],'post');
D = cat(3,D,d);
end

if dcatom == 1
    D(:,:,1) = padarray(1/M(1)*ones(M(1)),[s-M(1)  s-M(1)],'post');
end


D = single(D);
end