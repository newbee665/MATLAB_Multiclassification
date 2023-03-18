function P=panbie(thet,XE,ET)
%模型验证的应用
theta=thet';
    pai=exp(theta(1)+XE*theta(2:end))/(1+exp(theta(1)+XE*theta(2:end)));
    if(pai<=0.5)
        P=ET(1);
    else
        P=ET(2);
    end