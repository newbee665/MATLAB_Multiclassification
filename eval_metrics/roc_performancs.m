function [ prec, recall, fpr, AUCroc, Acc, F1_accuracy, F1_Precision,F1_Recall ,F1] = roc_performancs( labels, scores , plot_flag)
% EXAMPLE
%A：正确分类，判断为真，实际为真
%B：错误分类，判断为假，实际为真
%C：错误分类，判断为真，实际为假
%D：正确分类，判断为假，实际为假
% roc_performancs( [1 1 1 1 -1 -1 -1 -1]', [0.2 0.8 0.1 0.3 -0.1 -0.7 0.01 -0.05]' , 1)
%ROC空间将伪阳性率（FPR）定义为 X 轴，真阳性率（TPR）定义为 Y 轴。
%recall,召回率、查全率。也叫TPR真阳性率：A/(A+C)
%precision,准确率、正确率:A/(A+B)
%F测度：综合了查全率和准确率：F=2*P*R/(P+R)
%FPR伪阳性率：B/(B+D)

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
    xlabel('recall 召回率');
    ylabel('precision 精确率');
    grid on
    title('precision-recall PR曲线');
    
    subplot(1,2,2)
    plot(fpr, tpr, '-r', 'linewidth',2); % add pseudo point to complete curve
    xlabel('false positive rate 假阳率');
    ylabel('true positive rate 真阳率');
    grid on
    title('ROC curve ROC曲线');
end

AUCroc = 100*AUCroc; % Area Under the ROC curve
Acc = 100*sum(labels == sign(scores))/length(scores); % Accuracy


end

