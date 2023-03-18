function A=OVO(train,label,ET)
kind=length(unique(label));
fea=length(ET);
n=fea*(fea-1)/2;
arr=combntns(ET,2);
for i=1:kind
    class{i,1}=[train(label==i,:),label(label==i)];%Íê±¸¼¯
end
for i=1:n
    A{i}=[class{arr(i,1)};class{arr(i,2)}];
end
end
    
