clear all
wine_dataset = xlsread("C:\Users\ranab\Desktop\PhD_2\eece5644\hw1\winequality-white.csv");

data = wine_dataset(:, 1:end-1);
label = wine_dataset(:, end);

n = size(data, 2); % # of features 
N = size(data, 1); % # of samples
l = 11; 

alpha = 0.0000001;
C = zeros(n, n, l);
for i = 1:l
    C(:,:,i) = eye(n,n);
end

m = zeros(l, l);
lambda = zeros(1, l);
nc = zeros(1, l);
for i = 1:l
    if data(label==(i-1), :) ~= NaN
        m(:, i) = mean(data(label==(i-1), :)', 2);
        C(:, :, i) = cov(data(label==(i-1), :));
        lambda(i) = alpha.*trace(C(:, :, i))./rank(C(:, :, i));
        nc(i) = length(find(label==(i-1)));
    end
end

prior = nc/N;

for i = 1:l
    C(:, :, i) = C(:, :, i) + lambda(i).*eye(n, n); %Regularization
end


for i = 1:l
    classPost(i,:) = evalGaussian(data', m(:, i), C(:,:,i));
    classPost(i,:) = classPost(i,:)*prior(i);
end

[K, D] = max(classPost, [], 1);

[confusionM, classPriors] = confusionMatrix(label, D', 11, 11);


Perror = 1 - length(find(D' == label))/N;

save 'Q3.1.mat';


x_data_train = transpose(table2array(readtable("C:\Users\ranab\Desktop\PhD_2\eece5644\hw1\X_train.txt")));
y_data_train = transpose(table2array(readtable("C:\Users\ranab\Desktop\PhD_2\eece5644\hw1\y_train.txt")));
x_data_test = transpose(table2array(readtable("C:\Users\ranab\Desktop\PhD_2\eece5644\hw1\X_test.txt")));
y_data_test = transpose(table2array(readtable("C:\Users\ranab\Desktop\PhD_2\eece5644\hw1\y_test.txt")));

har_data = [x_data_train  x_data_test];


y_data = [y_data_train  y_data_test];

n_2 = size(har_data, 1); % # of features 
N_2 = size(har_data, 2); % # of samples
l_2 = 6; % # of labels

har_data = har_data';

alpha_2 = 0.0000001;
C_2 = zeros(n_2, n_2, l_2);
for i = 1:l_2
    C_2(:,:,i) = eye(n_2,n_2);
end

m_2 = zeros(n_2, l_2);
lambda_2 = zeros(1, l_2);
nc_2 = zeros(1, l_2);
for i = 1:l_2
    if har_data(y_data==i, :) ~= NaN
        m_2(:, i) = mean(har_data(y_data==i, :)', 2);
        C_2(:, :, i) = cov(har_data(y_data==i, :));
        lambda_2(i) = alpha_2.*trace(C_2(:, :, i))./rank(C_2(:, :, i));
        nc_2(i) = length(find(y_data==i));
    end
end


prior_2 = nc_2/N_2;

for i = 1:l_2
    C_2(:, :, i) = C_2(:, :, i) + lambda_2(i).*eye(n_2, n_2); %Regularization
end


for i = 1:l_2
    classPost_2(i,:) = evalGaussian(har_data', m_2(:, i), C_2(:,:,i));
    classPost_2(i,:) = classPost_2(i,:)*prior_2(i);
end

[K_2, D_2] = max(classPost_2, [], 1);

[confusionM_2, classPriors_2] = confusionMatrix(y_data', D_2', 6, 6);
Perror_2 = 1 - length(find(D_2' == y_data'))/N_2;


figure(1)

for i = 0:n-1
    idx = (label == i);
    plot3(data(idx, 2), data(idx, 1), data(idx, 3), '*');
    hold on
end
grid on
xlabel('X3');
ylabel('X1');
zlabel('X2');
legend;
title('Data Distribution');


figure(2)
for i = 0:n_2-1
    idx = (y_data == i);
    plot3(har_data(idx, 2), har_data(idx, 1), har_data(idx, 3), '*');
    hold on
end
grid on
xlabel('X3');
ylabel('X1');
zlabel('X2');
legend;
title('Data Distribution');



