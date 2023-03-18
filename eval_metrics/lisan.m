function A=lisan(X)
A=[];
[a,b]=size(X);
for i=1:a
    for j=1:b
    if X(i,j)<=2
        A(i,j)=1;
    elseif X(i,j)>2 && X(i,j)<=4
        A(i,j)=2;
    elseif X(i,j)>4 && X(i,j)<=6
        A(i,j)=3;
    elseif X(i,j)>6 && X(i,j)<=8
        A(i,j)=4;
    else
        A(i,j)=5;
    end
    end
end
