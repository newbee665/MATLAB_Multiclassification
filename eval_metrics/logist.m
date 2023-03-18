function theta=logist(X,Y0,ET)

n=size(Y0,1);
for i=1:n
    if Y0(i)==ET(1)
        Y1(i,1)=0.25;
    elseif Y0(i)==ET(2)
        Y1(i,1)=0.75;
    end
end
Y=log(Y1./(1-Y1));
b=regress(Y,X);
theta=b';
