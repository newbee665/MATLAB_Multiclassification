clear
clc
%% Parameter initialization
m_ant=50;
row=3;
S=1;
column=3;
r=0.1;
G=20;
Q=100;
D=rand(row,column);
n=size(D,1);
D(D==0)=inf;
eta=1./D;
tau=ones(n,n*row);
tabu={};
NC=1;
jbest={};
fbest={};
lbest=inf.*ones(G,1);
All = xlsread('xaa.xls');
fea = size(All,2)-1;
randIndex = randperm(size(All,1));
data = All(randIndex,:);
trainingrate = 0.8;
k = trainingrate*size(data,1);
traindata = data(1:k,:);
x_train = traindata(:,1:end-1);
Y_train = traindata(:,end);
test = data(k+1:end,:);
x_test = test(:,1:end-1);
Y_test = test(:,end);
Xtrain = mapminmax(x_train',0,10);
X1_train = Xtrain';
Xtest = mapminmax(x_test',0,10);
X1_test = Xtest';
X_train = lisan(X1_train);
X_test = lisan(X1_test);
%% Data decomposition
decom_t1 = OVR(X_train,Y_train);
decom_t2 = OVR(X1_train,Y_train);
decom_t3 = OVR(X1_test,Y_test);
%% Feature selection
addpath('./methods'); 
listFIFS = {'IG','chi-square','relief'};
selection_method = listFIFS{1}; 
numF = fea;

for i=1:m_ant
    tabu{i,1}=1;
    jbest{i,1}=1;
end
    Alit=2;
    tovisit=1;
for i=1:m_ant
    visited=tabu{i}(end);
    J=(find(eta(visited,:)~=0));
    P=J;
    for k=1:length(J)
        P(k)=(tau(visited,J(k))^S)*(eta(visited,J(k))^column);%选择概率
    end
    P=P/(sum(P));
    pcum=cumsum(P);
    Blit=rand;
    select=find(pcum>=Blit);
    XX=J(select(1));
    tovisit=ceil(XX/row);
    X1=mod(XX,row);
    if X1==0
        X=3;
    else
        X=X1;
    end
    if tovisit~=n
        tabu{i}(Alit)=tovisit;
        fbest{i,1}(Alit-1)=X;%运输方式矩阵
        jbest{i,1}(Alit)=XX;%精细节点
    end
end
for i=1:size(decom_t1,2)
  ranking_t1{i}=sumfeaselection(numF,selection_method,decom_t1{1,i}(:,1:end-1) ,decom_t1{1,i}(:,end));
  K=5;
  Fea_t1{i}=ranking_t1{i}(1:K);
  decom_t4=decom_t2{i}(:,Fea_t1{i});
  decom_t5{i}=decom_t3{i}(:,Fea_t1{i});
  GM{i}=fitglm(decom_t4,decom_t2{i}(:,end),'Distribution','binomial','Link','logit');
end
%% Discriminating Scenarios
for i=1:size(x_test,1)
    for j=1:size(decom_t5,2)
        test_t1(i,j)=predict(GM{j},decom_t5{j}(i,:));
        if test_t1(i,j)>0.5
            test_t1(i,j)=1;
        else
            test_t1(i,j)=0;
        end
    end
    ET=find(test_t1(i,:)==1);
    if isempty(ET)==1
        kind(i,1)=4;
    elseif size(ET,2)==1
        kind(i,1)=ET;
    else
        decom_t6=OVO(X_train,Y_train,ET);
        decom_t7=OVO(X1_train,Y_train,ET);
        ranking_t2= sumfeaselection(numF,selection_method,decom_t6{1}(:,1:end-1),decom_t6{1}(:,end));
        Fea_t2=ranking_t2(1:K);
        decom_t8=[ones(size(decom_t7{1},1),1),decom_t7{1}(:,Fea_t2)];
        theta=logist(decom_t8,decom_t7{1}(:,end),ET);
        decom_t10=decom_t3{1}(:,Fea_t2);
        kind(i,1)=panbie(theta,decom_t10(i,:),ET);
    end
end

for i=1:size(GM,2)
    scores = GM{1,i}.Fitted.Probability; 
    [prec{i},tpr{i},fpr{i}, AUCroc(i), ~, ~, ~, ~,F1] = roc_performancs(decom_t1{1,i}(:,end), scores ,0);
end
    Accuracy = AUCarea(tpr,fpr,AUCroc)
    Stability = kuncheva_stability(tpr,fpr,AUCroc)
    



