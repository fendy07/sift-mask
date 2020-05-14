clear
clc

%load feature data after preprocessing with SIFT
load('feature.mat'); 
%mengetahui baris dan kolom dalam fitur
%get row and columns in feature data
[row, col] =size(feature_all); 

label = feature_all(:,col);
feature = feature_all(:,1:col-1);

%Split the data with Cross Validation
[test,train] = crossvalind('HoldOut',label,0.7); 


[row_train,col_train] =size(train);
temp_train=0
temp_test=0
for i=1:row_train
    if train(i,:) == 1
        temp_train=temp_train+1;
        feature_train(temp_train,:) = feature(i,:);
        label_train(temp_train,:) = label(i,:);
    
    else
        temp_test=temp_test+1;
        feature_test(temp_test,:) = feature(i,:);
        label_test(temp_test,:) = label(i,:);
    end
end

%pembuatan model 
%make model classification with K-NN
%knn
model_knn_chebychev = fitcknn(feature_train,label_train,'Distance','chebychev'); %pembuatan model kernel knn
model_knn_minkowski = fitcknn(feature_train,label_train, 'Distance','minkowski');
model_knn_euclidean = fitcknn(feature_train,label_train, 'Distance','euclidean');
loss_knn_chebychev = loss(model_knn_chebychev,feature_train, label_train, 'LossFun','hinge');
loss_knn_minskowski = loss(model_knn_minkowski,feature_train, label_train, 'LossFun','hinge');
loss_knn_euclidean = loss(model_knn_euclidean,feature_train, label_train, 'LossFun', 'hinge');

%svm kernel
t_gaussian=templateSVM('KernelFunction','gaussian');
t_rbf=templateSVM('KernelFunction','rbf');
t_linear=templateSVM('KernelFunction','linear');

%onevsone
model_svm_onevsone_gaussian=fitcecoc(feature_train, label_train,'Learners',t_gaussian);
model_svm_onevsone_rbf=fitcecoc(feature_train, label_train,'Learners',t_rbf);
model_svm_onevsone_linear=fitcecoc(feature_train, label_train,'Learners',t_linear);

%onevsall
model_svm_onevsall_gaussian=fitcecoc(feature_train, label_train,'Learners',t_gaussian,'Coding','onevsall');
model_svm_onevsall_rbf=fitcecoc(feature_train, label_train,'Learners',t_rbf,'Coding','onevsall');
model_svm_onevsall_linear=fitcecoc(feature_train, label_train,'Learners',t_linear,'Coding','onevsall');

%random forest
model_randomforest = TreeBagger(100,feature_train, label_train);

%testing process model predict for the data
%knn
%distance metric chebychev
result_knn_chebychev = predict(model_knn_chebychev,feature_test);
counter_knn_chebyshev = 0;
[row_test,col_test] = size(feature_test);
for i=1:row_test
    if result_knn_chebychev(i)==label_test(i);
        counter_knn_chebyshev=counter_knn_chebyshev+1;
    end
end
accuracy_knn_chebychev=(counter_knn_chebyshev/row_test)*100;
conf_mat_knn_chebychev=confusionmat(label_test,result_knn_chebychev);

%knn
%distance metric minkowski
result_knn_minkowski = predict(model_knn_minkowski,feature_test);
counter_knn_minkowski = 0;
for i=1:row_test
    if result_knn_minkowski(i)==label_test(i)
        counter_knn_minkowski=counter_knn_minkowski+1;
    end
end
accuracy_knn_minkowski=(counter_knn_minkowski/row_test)*100;
conf_mat_knn_minkowski=confusionmat(label_test,result_knn_minkowski);

%knn
%distance metric euclidean
result_knn_euclidean = predict(model_knn_euclidean,feature_test);
counter_knn_euclidean = 0;
for i=1:row_test
    if result_knn_euclidean(i)==label_test(i)
        counter_knn_euclidean=counter_knn_euclidean+1;
    end
end
accuracy_knn_euclidean=(counter_knn_euclidean/row_test)*100;
conf_mat_knn_euclidean=confusionmat(label_test,result_knn_euclidean);

%random forest
%distance metric rand forest
result_randomforest_randforest = predict(model_randomforest,feature_test);
result_randomforest_randforest=str2num(cell2mat(result_randomforest_randforest));
counter_randomforest_randforest = 0;
[row_test,col_test] = size(feature_test);
for i=1:row_test
    if result_randomforest_randforest(i)==label_test(i);
        counter_randomforest_randforest=counter_randomforest_randforest+1;
    end
end
accuracy_randomforest_randforest=(counter_randomforest_randforest/row_test)*100;
conf_mat_randomforest_randforest=confusionmat(label_test,result_randomforest_randforest);

%svm
%distance metric One Vs One
%Gaussian
result_svm_onevsone_gaussian = predict(model_svm_onevsone_gaussian,feature_test);
counter_svm_onevsone_gaussian = 0;
for i=1:row_test
    if result_svm_onevsone_gaussian(i)==label_test(i)
        counter_svm_onevsone_gaussian=counter_svm_onevsone_gaussian+1;
    end
end
accuracy_svm_onevsone_gaussian=(counter_svm_onevsone_gaussian/row_test)*100;
conf_mat_svm_onevsone_gaussian=confusionmat(label_test,result_svm_onevsone_gaussian);

%svm
%Distance Metric One Vs All
%Gaussian
result_svm_onevsall_gaussian = predict(model_svm_onevsall_gaussian,feature_test);
counter_svm_onevsall_gaussian = 0;
for i=1:row_test
    if result_svm_onevsall_gaussian(i)==label_test(i)
        counter_svm_onevsall_gaussian=counter_svm_onevsall_gaussian+1;
    end
end
accuracy_svm_onevsall_gaussian=(counter_svm_onevsall_gaussian/row_test)*100;
conf_mat_svm_onevsall_gaussian=confusionmat(label_test,result_svm_onevsall_gaussian);

%SVM
%One vs All Linear
result_svm_onevsall_linear = predict(model_svm_onevsall_linear,feature_test);
counter_svm_onevsall_linear = 0;
for i=1:row_test
    if result_svm_onevsall_linear(i)==label_test(i)
        counter_svm_onevsall_linear=counter_svm_onevsall_linear+1;
    end
end
accuracy_svm_onevsall_linear=(counter_svm_onevsall_linear/row_test)*100;
conf_mat_svm_onevsall_linear=confusionmat(label_test,result_svm_onevsall_linear);

%One vs One Linear
result_svm_onevsone_linear = predict(model_svm_onevsone_linear,feature_test);
counter_svm_onevsone_linear = 0;
for i=1:row_test
    if result_svm_onevsone_linear(i)==label_test(i)
        counter_svm_onevsone_linear=counter_svm_onevsone_linear+1;
    end
end
accuracy_svm_onevsone_linear=(counter_svm_onevsone_linear/row_test)*100;
conf_mat_svm_onevsone_linear=confusionmat(label_test,result_svm_onevsone_linear);

%Radial Basis Function(RBF)
result_svm_onevsall_rbf = predict(model_svm_onevsall_rbf,feature_test);
counter_svm_onevsall_rbf = 0;
for i=1:row_test
    if result_svm_onevsall_rbf(i)==label_test(i)
        counter_svm_onevsall_rbf=counter_svm_onevsall_rbf+1;
    end
end
accuracy_svm_onevsall_rbf=(counter_svm_onevsall_rbf/row_test)*100;
conf_mat_svm_onevsall_rbf=confusionmat(label_test,result_svm_onevsall_rbf);

%Radial Basis Function(RBF)
result_svm_onevsone_rbf = predict(model_svm_onevsone_rbf,feature_test);
counter_svm_onevsone_rbf = 0;
for i=1:row_test
    if result_svm_onevsone_rbf(i)==label_test(i)
        counter_svm_onevsone_rbf=counter_svm_onevsone_rbf+1;
    end
end
accuracy_svm_onevsone_rbf=(counter_svm_onevsone_rbf/row_test)*100;
conf_mat_svm_onevsone_rbf=confusionmat(label_test,result_svm_onevsone_rbf);
