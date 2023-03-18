function A=OVR(train,label)
fea=length(unique(label));
num=size(train,1);
for i=1:fea
    temp=label;
   for j=1:num
    if temp(j)~=i
        temp(j)=0;
    end
    if temp(j)~=0
        temp(j)=1;
    end   
   end
   A{i}=[train,temp];
   temp=[];
end
end