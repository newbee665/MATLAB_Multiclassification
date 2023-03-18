function S = kuncheva_stability(tpr,fpr,AUCroc)

k = size(tpr,2);
q = size(fpr,2);

r = NaN(q,q);
% kuncheva index r
for n = 1:q-1
    for m = n+1:q
        r(n,m) = length(intersect(AUCroc(:,n),AUCroc(:,m)));
    end
end
d=q-1;
A = (r-(k^2/d))./(k-(k^2/d));
A(isnan(A)) = 0;
S1 = 2.*sum(A(:))./(q*(q-1));
for i=1:k
    arc=find(tpr{1,i}==max(tpr{1,i}));
    tpr{1,i}(arc)=0;
    pir(i)=max(tpr{1,i});
end
S = mean(pir);
