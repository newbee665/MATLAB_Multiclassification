function [ prec, recall, fpr, AUCroc, Acc, F1_accuracy, F1_Precision,F1_Recall ,F1] = roc_performancs( labels, scores , plot_flag)
% EXAMPLE
%A����ȷ���࣬�ж�Ϊ�棬ʵ��Ϊ��
%B��������࣬�ж�Ϊ�٣�ʵ��Ϊ��
%C��������࣬�ж�Ϊ�棬ʵ��Ϊ��
%D����ȷ���࣬�ж�Ϊ�٣�ʵ��Ϊ��
% roc_performancs( [1 1 1 1 -1 -1 -1 -1]', [0.2 0.8 0.1 0.3 -0.1 -0.7 0.01 -0.05]' , 1)
%ROC�ռ佫α�����ʣ�FPR������Ϊ X �ᣬ�������ʣ�TPR������Ϊ Y �ᡣ
%recall,�ٻ��ʡ���ȫ�ʡ�Ҳ��TPR�������ʣ�A/(A+C)
%precision,׼ȷ�ʡ���ȷ��:A/(A+B)
%F��ȣ��ۺ��˲�ȫ�ʺ�׼ȷ�ʣ�F=2*P*R/(P+R)
%FPRα�����ʣ�B/(B+D)

[Xfpr,Ytpr,~,AUCroc]  = perfcurve(double(labels), double(scores), 1,'TVals','all','xCrit', 'fpr', 'yCrit', 'tpr');
[Xpr,Ypr,~,AUCpr] = perfcurve(double(labels), double(scores), 1, 'TVals','all','xCrit', 'reca', 'yCrit', 'prec');
[acc,~,~,~] = perfcurve(double(labels), double(scores), 1,'xCrit', 'accu');

prec = Ypr; 
prec(isnan(prec))=1;
tpr = Ytpr; 
tpr(isnan(tpr))=0;% recall = true positive rate
fpr = Xfpr; % (1 - Specificity)
recall = tpr;

% Compute F-Measure
f1= 2*(prec.*tpr) ./ (prec+tpr);
[Max_F1,idx] = max(f1);
F1_Precision = prec(idx);
F1_Recall = tpr(idx);
F1_accuracy = acc(idx);
F1=Max_F1;


if plot_flag
    figure;
    subplot(1,2,1)
    plot(tpr,  prec, '-b', 'linewidth',2); % add pseudo point to complete curve
    xlabel('recall �ٻ���');
    ylabel('precision ��ȷ��');
    grid on
    title('precision-recall PR����');
    
    subplot(1,2,2)
    plot(fpr, tpr, '-r', 'linewidth',2); % add pseudo point to complete curve
    xlabel('false positive rate ������');
    ylabel('true positive rate ������');
    grid on
    title('ROC curve ROC����');
end

AUCroc = 100*AUCroc; % Area Under the ROC curve
Acc = 100*sum(labels == sign(scores))/length(scores); % Accuracy


end

