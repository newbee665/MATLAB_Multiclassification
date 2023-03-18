function P = AUCarea(tpr,fpr,roc)
[a,b]=size(fpr);
for i=1:a
    tpr{1,a}=mean(sum(roc(a)+roc(a+1))+roc(1)*roc(end));
end
for i=1:b
    arr=find(fpr{1,i}==max(fpr{1,i}));
    fpr{1,i}(arr)=0;
    pic(i)=max(fpr{1,i});
end
P=mean(pic);