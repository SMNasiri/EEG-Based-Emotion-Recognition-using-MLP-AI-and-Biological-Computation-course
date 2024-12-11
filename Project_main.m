%% section 1
clear, clc

data = load('Project_data.mat');

%% section 2

train_data = data.TrainData;
train_labels = data.TrainLabels;
fs = data.fs;
test_data = data.TestData;

%% section 3.1
%Features

num_data = size(train_data,3);
num_channel = size(train_data,1);

variance = zeros(num_data,num_channel);
covariance = zeros(num_data,num_channel,num_channel);
FF = zeros(num_data,num_channel);

f_max = zeros(num_data,num_channel);
f_mean = zeros(num_data,num_channel);
f_med = zeros(num_data,num_channel);
f_ratio1 = zeros(num_data,num_channel);
f_ratio2 = zeros(num_data,num_channel);
f_ratio3 = zeros(num_data,num_channel);
f_ratio4 = zeros(num_data,num_channel);
f_ratio5 = zeros(num_data,num_channel);
f_ratio6 = zeros(num_data,num_channel);
f_ratio7 = zeros(num_data,num_channel);

for i=1:num_data
    sample = train_data(:,:,i).';
    
    variance(i,:) = var(sample);
    covariance(i,:,:) = cov(sample);
    
    d = diff(sample);
    dv1 = std(d);
    dd = diff(d);
    dv2 = std(dd);
    v = std(sample);
    FF(i,:) = v.*dv2./dv1./dv1;
    
    % Calculate the PSD
    [pxx,f] = pwelch(sample,[],[],[],fs);
    % Find the frequency with maximum abundance
    [max_val, max_idx] = max(pxx);
    f_max(i,:) = f(max_idx).';
    
    f_mean(i,:) = meanfreq(sample,fs);
    f_med(i,:) = medfreq(sample,fs);
    
    fbandt = bandpower(sample);
    fband1 = bandpower(sample,fs,[0.1 3]);
    f_ratio1(i,:) = fband1./fbandt;
    fband2 = bandpower(sample,fs,[4 7]);
    f_ratio2(i,:) = fband2./fbandt;
    fband3 = bandpower(sample,fs,[8 12]);
    f_ratio3(i,:) = fband3./fbandt;
    fband4 = bandpower(sample,fs,[12 15]);
    f_ratio4(i,:) = fband4./fbandt;
    fband5 = bandpower(sample,fs,[16 20]);
    f_ratio5(i,:) = fband5./fbandt;
    fband6 = bandpower(sample,fs,[21 30]);
    f_ratio6(i,:) = fband6./fbandt;
    fband7 = bandpower(sample,fs,[30 100]);
    f_ratio7(i,:) = fband7./fbandt;
end

%% section 3.2
%normalize

variance = normalize(variance);
for i=1:num_channel
    for j=1:num_channel
        covariance(:,i,j) = normalize(covariance(:,i,j));
    end 
end
FF = normalize(FF);

f_max = normalize(f_max);
f_mean = normalize(f_mean);
f_med = normalize(f_med);
f_ratio1 = normalize(f_ratio1);
f_ratio2 = normalize(f_ratio2);
f_ratio3 = normalize(f_ratio3);
f_ratio4 = normalize(f_ratio4);
f_ratio5 = normalize(f_ratio5);
f_ratio6 = normalize(f_ratio6);
f_ratio7 = normalize(f_ratio7);

%% section 4.1
%feature selection

labels = train_labels.';

variance_selection = zeros(1,num_channel);
covariance_selection = zeros(1,num_channel,num_channel);
FF_selection = zeros(1,num_channel);

f_max_selection = zeros(1,num_channel);
f_mean_selection = zeros(1,num_channel);
f_med_selection = zeros(1,num_channel);
f_ratio1_selection = zeros(1,num_channel);
f_ratio2_selection = zeros(1,num_channel);
f_ratio3_selection = zeros(1,num_channel);
f_ratio4_selection = zeros(1,num_channel);
f_ratio5_selection = zeros(1,num_channel);
f_ratio6_selection = zeros(1,num_channel);
f_ratio7_selection = zeros(1,num_channel);

for i=1:num_channel
    mu_0 = mean(variance(:,i));
    mu_1 = mean(variance((labels == 1),i));
    mu_minus_1 = mean(variance((labels == -1),i));
    v_1 = var(variance((labels == 1),i));
    v_minus_1 = var(variance((labels == -1),i));
    variance_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    for j=(i+1):num_channel
        mu_0 = mean(covariance(:,i,j));
        mu_1 = mean(covariance((labels == 1),i,j));
        mu_minus_1 = mean(covariance((labels == -1),i,j));
        v_1 = var(covariance((labels == 1),i,j));
        v_minus_1 = var(covariance((labels == -1),i,j));
        covariance_selection(1,i,j) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
    end
end

for i=1:num_channel
    mu_0 = mean(FF(:,i));
    mu_1 = mean(FF((labels == 1),i));
    mu_minus_1 = mean(FF((labels == -1),i));
    v_1 = var(FF((labels == 1),i));
    v_minus_1 = var(FF((labels == -1),i));
    FF_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_max(:,i));
    mu_1 = mean(f_max((labels == 1),i));
    mu_minus_1 = mean(f_max((labels == -1),i));
    v_1 = var(f_max((labels == 1),i));
    v_minus_1 = var(f_max((labels == -1),i));
    f_max_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_mean(:,i));
    mu_1 = mean(f_mean((labels == 1),i));
    mu_minus_1 = mean(f_mean((labels == -1),i));
    v_1 = var(f_mean((labels == 1),i));
    v_minus_1 = var(f_mean((labels == -1),i));
    f_mean_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_med(:,i));
    mu_1 = mean(f_med((labels == 1),i));
    mu_minus_1 = mean(f_med((labels == -1),i));
    v_1 = var(f_med((labels == 1),i));
    v_minus_1 = var(f_med((labels == -1),i));
    f_med_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio1(:,i));
    mu_1 = mean(f_ratio1((labels == 1),i));
    mu_minus_1 = mean(f_ratio1((labels == -1),i));
    v_1 = var(f_ratio1((labels == 1),i));
    v_minus_1 = var(f_ratio1((labels == -1),i));
    f_ratio1_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio2(:,i));
    mu_1 = mean(f_ratio2((labels == 1),i));
    mu_minus_1 = mean(f_ratio2((labels == -1),i));
    v_1 = var(f_ratio2((labels == 1),i));
    v_minus_1 = var(f_ratio2((labels == -1),i));
    f_ratio2_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio3(:,i));
    mu_1 = mean(f_ratio3((labels == 1),i));
    mu_minus_1 = mean(f_ratio3((labels == -1),i));
    v_1 = var(f_ratio3((labels == 1),i));
    v_minus_1 = var(f_ratio3((labels == -1),i));
    f_ratio3_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio4(:,i));
    mu_1 = mean(f_ratio4((labels == 1),i));
    mu_minus_1 = mean(f_ratio4((labels == -1),i));
    v_1 = var(f_ratio4((labels == 1),i));
    v_minus_1 = var(f_ratio4((labels == -1),i));
    f_ratio4_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio5(:,i));
    mu_1 = mean(f_ratio5((labels == 1),i));
    mu_minus_1 = mean(f_ratio5((labels == -1),i));
    v_1 = var(f_ratio5((labels == 1),i));
    v_minus_1 = var(f_ratio5((labels == -1),i));
    f_ratio5_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio6(:,i));
    mu_1 = mean(f_ratio6((labels == 1),i));
    mu_minus_1 = mean(f_ratio6((labels == -1),i));
    v_1 = var(f_ratio6((labels == 1),i));
    v_minus_1 = var(f_ratio6((labels == -1),i));
    f_ratio6_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

for i=1:num_channel
    mu_0 = mean(f_ratio7(:,i));
    mu_1 = mean(f_ratio7((labels == 1),i));
    mu_minus_1 = mean(f_ratio7((labels == -1),i));
    v_1 = var(f_ratio7((labels == 1),i));
    v_minus_1 = var(f_ratio7((labels == -1),i));
    f_ratio7_selection(1,i) = ((mu_1 - mu_0)^2 + (mu_minus_1 - mu_0)^2)/(v_1 + v_minus_1);
end

%% section 4.2

l_stat = num_channel + num_channel*(num_channel-1)/2 + num_channel;
feature_stat = zeros(1,l_stat);

for i=1:l_stat
    if (i<=num_channel)
        feature_stat(1,i) = variance_selection(1,i);
    end
    if i == (num_channel + 1)
        t = i;
        for j=1:(num_channel-1)
            feature_stat(1,t:(t + num_channel-j-1)) = covariance_selection(1,j,(j+1):num_channel);
            t = t + num_channel-j;
        end
    end
    if (i>num_channel + num_channel*(num_channel-1)/2)
        feature_stat(1,i) = FF_selection(1,i-(num_channel + num_channel*(num_channel-1)/2));
    end
end
        
l_freq = 10*num_channel;
feature_freq = zeros(1,l_freq);

for i=1:l_freq
    if (i<=num_channel)
        feature_freq(1,i) = f_max_selection(1,i);
    end
    if i>num_channel && i<=(2*num_channel)
        feature_freq(1,i) = f_mean_selection(1,i-num_channel);
    end
    if i>2*num_channel && i<=(3*num_channel)
        feature_freq(1,i) = f_med_selection(1,i-2*num_channel);
    end
    if i>3*num_channel && i<=(4*num_channel)
        feature_freq(1,i) = f_ratio1_selection(1,i-3*num_channel);
    end
    if i>4*num_channel && i<=(4*num_channel)
        feature_freq(1,i) = f_ratio2_selection(1,i-4*num_channel);
    end
    if i>5*num_channel && i<=(6*num_channel)
        feature_freq(1,i) = f_ratio3_selection(1,i-5*num_channel);
    end
    if i>6*num_channel && i<=(7*num_channel)
        feature_freq(1,i) = f_ratio4_selection(1,i-6*num_channel);
    end
    if i>7*num_channel && i<=(8*num_channel)
        feature_freq(1,i) = f_ratio5_selection(1,i-7*num_channel);
    end
    if i>8*num_channel && i<=(9*num_channel)
        feature_freq(1,i) = f_ratio6_selection(1,i-8*num_channel);
    end
    if i>9*num_channel
        feature_freq(1,i) = f_ratio7_selection(1,i-9*num_channel);
    end
end

%% section 4.3

[sorted_feature_stat, idx] = sort(feature_stat, 'descend');
max_values_stat = sorted_feature_stat(1:30);
max_indices_stat = idx(1:30);

[sorted_feature_freq, idx] = sort(feature_freq, 'descend');
max_values_freq = sorted_feature_freq(1:20);
max_indices_freq = idx(1:20);

%% section 4.4

best_feature = zeros(num_data,50);

variance_selected = zeros(1,num_channel);
covariance_selected = zeros(num_channel,num_channel);
FF_selected = zeros(1,num_channel);

f_max_selected = zeros(1,num_channel);
f_mean_selected = zeros(1,num_channel);
f_med_selected = zeros(1,num_channel);
f_ratio1_selected = zeros(1,num_channel);
f_ratio2_selected = zeros(1,num_channel);
f_ratio3_selected = zeros(1,num_channel);
f_ratio4_selected = zeros(1,num_channel);
f_ratio5_selected = zeros(1,num_channel);
f_ratio6_selected = zeros(1,num_channel);
f_ratio7_selected = zeros(1,num_channel);

for i=1:30
    l = max_indices_stat(i);
    if (l<=num_channel)
        best_feature(:,i) = variance(:,l);
        variance_selected(1,l) = 1;
    end
    if l>num_channel && l<= (num_channel + num_channel*(num_channel-1)/2)
        t = l - num_channel;
        for j = 1:(num_channel-1)
            if t<=(num_channel-j) && t>0
                best_feature(:,i) = covariance(:,j,t + j);
                covariance_selected(j, t + j) = 1;
                t = 0;
            else
                t = t - (num_channel-j);
            end
        end
    end
    if (l>(num_channel + num_channel*(num_channel-1)/2))
        best_feature(:,i) = FF(:,(l - (num_channel + num_channel*(num_channel-1)/2)));
        FF_selected(1,(l - (num_channel + num_channel*(num_channel-1)/2))) = 1;
    end
end

for i=1:20
    l = max_indices_freq(i);
    if (l<=num_channel)
        best_feature(:,(i+30)) = f_max(:,l);
        f_max_selected(1,l) = 1;
    end
    if l>num_channel && l<=(2*num_channel)
        best_feature(:,(i+30)) = f_mean(:,l-num_channel);
        f_mean_selected(1,l-num_channel) = 1;
    end
    if l>2*num_channel && l<=(3*num_channel)
        best_feature(:,(i+30)) = f_med(:,l-2*num_channel);
        f_med_selected(1,l-2*num_channel) = 1;
    end
    if l>3*num_channel && l<=(4*num_channel)
        best_feature(:,(i+30)) = f_ratio1(:,l-3*num_channel);
        f_ratio1_selected(1,l-3*num_channel) = 1;
    end
    if l>4*num_channel && l<=(5*num_channel)
        best_feature(:,(i+30)) = f_ratio2(:,l-4*num_channel);
        f_ratio2_selected(1,l-4*num_channel) = 1;
    end
    if l>5*num_channel && l<=(6*num_channel)
        best_feature(:,(i+30)) = f_ratio3(:,l-5*num_channel);
        f_ratio3_selected(1,l-5*num_channel) = 1;
    end
    if l>6*num_channel && l<=(7*num_channel)
        best_feature(:,(i+30)) = f_ratio4(:,l-6*num_channel);
        f_ratio4_selected(1,l-6*num_channel) = 1;
    end
    if l>7*num_channel && l<=(8*num_channel)
        best_feature(:,(i+30)) = f_ratio5(:,l-7*num_channel);
        f_ratio5_selected(1,l-7*num_channel) = 1;
    end
    if l>8*num_channel && l<=(9*num_channel)
        best_feature(:,(i+30)) = f_ratio6(:,l-8*num_channel);
        f_ratio6_selected(1,l-8*num_channel) = 1;
    end
    if l>9*num_channel
        best_feature(:,(i+30)) = f_ratio7(:,l-9*num_channel);
        f_ratio7_selected(1,l-9*num_channel) = 1;
    end
end

%% section 5.1
%MLP

n = 25;
m = 25;
accuracy_MLP = zeros(n,m);
k = 5;

for i=1:n
    for j = 1:m
        acc = 0;
        for l = 1:5
            c = cvpartition(num_data,'KFold',k);
            net = fitnet([i j]);
            for t = 1:k
                train_idx = c.training(t);
                test_idx = c.test(t);

                X_train = best_feature(train_idx,:).';
                y_train = train_labels(:,train_idx);
                X_val = best_feature(test_idx,:).';
                y_val = train_labels(:,test_idx);
    
                net = train(net, X_train, y_train);
    
                y_pred = (sim(net, X_val)>0);
                y_pred = 2*y_pred - 1;
    
                a = sum(y_pred == y_val);
                acc = acc + a/size(y_val,2);
            end
        end
        accuracy_MLP(i,j) = acc/5/k;
    end
end

%% section 5.2

% Find the maximum value and its index
[max_value, max_idx] = max(accuracy_MLP(:));

% Convert the index to row and column subscripts
[max_row, max_col] = ind2sub(size(accuracy_MLP), max_idx);

num_layer1 = max_row;
num_layer2 = max_col;

%% section 5.3

c = cvpartition(num_data,'KFold',k);
net = fitnet([num_layer1 num_layer2]);

train_idx = c.training(1);
test_idx = c.test(1);

X_train = best_feature(train_idx,:).';
y_train = train_labels(:,train_idx);
X_val = best_feature(test_idx,:).';
y_val = train_labels(:,test_idx);
    
net = train(net, X_train, y_train);
    
y_pred = (sim(net, X_val)>0);
y_pred = 2*y_pred - 1;
    
a = sum(y_pred == y_val);
acc = a/size(y_val,2);

%% section 5.4

num_data_test = size(test_data,3);
num_channel_test = size(test_data,1);

variance_test = zeros(num_data_test,num_channel_test);
covariance_test = zeros(num_data_test,num_channel_test,num_channel_test);
FF_test = zeros(num_data_test,num_channel_test);

f_max_test = zeros(num_data_test,num_channel_test);
f_mean_test = zeros(num_data_test,num_channel_test);
f_med_test = zeros(num_data_test,num_channel_test);
f_ratio1_test = zeros(num_data_test,num_channel_test);
f_ratio2_test = zeros(num_data_test,num_channel_test);
f_ratio3_test = zeros(num_data_test,num_channel_test);
f_ratio4_test = zeros(num_data_test,num_channel_test);
f_ratio5_test = zeros(num_data_test,num_channel_test);
f_ratio6_test = zeros(num_data_test,num_channel_test);
f_ratio7_test = zeros(num_data_test,num_channel_test);

for i=1:num_data_test
    sample = test_data(:,:,i).';
    
    variance_test(i,:) = var(sample);
    covariance_test(i,:,:) = cov(sample);
    
    d = diff(sample);
    dv1 = std(d);
    dd = diff(d);
    dv2 = std(dd);
    v = std(sample);
    FF_test(i,:) = v.*dv2./dv1./dv1;
    
    % Calculate the PSD
    [pxx,f] = pwelch(sample,[],[],[],fs);
    % Find the frequency with maximum abundance
    [max_val, max_idx] = max(pxx);
    f_max_test(i,:) = f(max_idx).';
    
    f_mean_test(i,:) = meanfreq(sample,fs);
    f_med_test(i,:) = medfreq(sample,fs);
    
    fbandt = bandpower(sample);
    fband1 = bandpower(sample,fs,[0.1 3]);
    f_ratio1_test(i,:) = fband1./fbandt;
    fband2 = bandpower(sample,fs,[4 7]);
    f_ratio2_test(i,:) = fband2./fbandt;
    fband3 = bandpower(sample,fs,[8 12]);
    f_ratio3_test(i,:) = fband3./fbandt;
    fband4 = bandpower(sample,fs,[12 15]);
    f_ratio4_test(i,:) = fband4./fbandt;
    fband5 = bandpower(sample,fs,[16 20]);
    f_ratio5_test(i,:) = fband5./fbandt;
    fband6 = bandpower(sample,fs,[21 30]);
    f_ratio6_test(i,:) = fband6./fbandt;
    fband7 = bandpower(sample,fs,[30 100]);
    f_ratio7_test(i,:) = fband7./fbandt;
end

variance_test = normalize(variance_test);
for i=1:num_channel_test
    for j=1:num_channel_test
        covariance_test(:,i,j) = normalize(covariance_test(:,i,j));
    end 
end
FF_test = normalize(FF_test);

f_max_test = normalize(f_max_test);
f_mean_test = normalize(f_mean_test);
f_med_test = normalize(f_med_test);
f_ratio1_test = normalize(f_ratio1_test);
f_ratio2_test = normalize(f_ratio2_test);
f_ratio3_test = normalize(f_ratio3_test);
f_ratio4_test = normalize(f_ratio4_test);
f_ratio5_test = normalize(f_ratio5_test);
f_ratio6_test = normalize(f_ratio6_test);
f_ratio7_test = normalize(f_ratio7_test);

best_feature_test = zeros(num_data_test,50);

for i=1:30
    l = max_indices_stat(i);
    if (l<=num_channel_test)
        best_feature_test(:,i) = variance_test(:,l);
    end
    if l>num_channel_test && l<= (num_channel_test + num_channel_test*(num_channel_test-1)/2)
        t = l - num_channel_test;
        for j = 1:(num_channel_test-1)
            if t<=(num_channel_test-j) && t>0
                best_feature_test(:,i) = covariance_test(:,j,t + j);
                t = 0;
            else
                t = t - (num_channel_test-j);
            end
        end
    end
    if (l>(num_channel_test + num_channel_test*(num_channel_test-1)/2))
        best_feature_test(:,i) = FF_test(:,(l - (num_channel_test + num_channel_test*(num_channel_test-1)/2)));
    end
end

for i=1:20
    l = max_indices_freq(i);
    if (l<=num_channel_test)
        best_feature_test(:,(i+30)) = f_max_test(:,l);
    end
    if l>num_channel_test && l<=(2*num_channel_test)
        best_feature_test(:,(i+30)) = f_mean_test(:,l-num_channel_test);
    end
    if l>2*num_channel_test && l<=(3*num_channel_test)
        best_feature_test(:,(i+30)) = f_med_test(:,l-2*num_channel_test);
    end
    if l>3*num_channel_test && l<=(4*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio1_test(:,l-3*num_channel_test);
    end
    if l>4*num_channel_test && l<=(5*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio2_test(:,l-4*num_channel_test);
    end
    if l>5*num_channel_test && l<=(6*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio3_test(:,l-5*num_channel_test);
    end
    if l>6*num_channel_test && l<=(7*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio4_test(:,l-6*num_channel_test);
    end
    if l>7*num_channel_test && l<=(8*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio5_test(:,l-7*num_channel_test);
    end
    if l>8*num_channel_test && l<=(9*num_channel_test)
        best_feature_test(:,(i+30)) = f_ratio6_test(:,l-8*num_channel_test);
    end
    if l>9*num_channel_test
        best_feature_test(:,(i+30)) = f_ratio7_test(:,l-9*num_channel_test);
    end
end

%% section 5.5

X_test = best_feature_test.';

y_test_pred_MLP = (sim(net, X_test)>0);
y_test_pred_MLP = 2*y_test_pred_MLP - 1;

save('labels_test_predicted_MLP.mat', 'y_test_pred_MLP');

%% section 6.1
%RBF

n = 1:40;
m = 0.1:0.1:4;
accuracy_RBF = zeros(size(n,2),size(m,2));
k = 5;

for i=n
    for j=1:size(m,2)
        acc = 0;
        for l=1:5
            c = cvpartition(num_data,'KFold',k);
            for t = 1:k
                train_idx = c.training(t);
                test_idx = c.test(t);

                X_train = best_feature(train_idx,:).';
                y_train = train_labels(:,train_idx);
                X_val = best_feature(test_idx,:).';
                y_val = train_labels(:,test_idx);
    
                net = newrb(X_train,y_train, 0, m(j), i);
    
                y_pred = (sim(net, X_val)>0);
                y_pred = 2*y_pred - 1;
    
                a = sum(y_pred == y_val);
                acc = acc + a/size(y_val,2);
            end
        end
        accuracy_RBF(i,j) = acc/5/k;
    end
end

%% section 6.2

% Find the maximum value and its index
[max_value, max_idx] = max(accuracy_RBF(:));

% Convert the index to row and column subscripts
[max_row, max_col] = ind2sub(size(accuracy_RBF), max_idx);

num_neurons = max_row;
Spread = m(max_col);

%% section 6.3

c = cvpartition(num_data,'KFold',k);

train_idx = c.training(1);
test_idx = c.test(1);

X_train = best_feature(train_idx,:).';
y_train = train_labels(:,train_idx);
X_val = best_feature(test_idx,:).';
y_val = train_labels(:,test_idx);
    
net = newrb(X_train,y_train, 0, Spread, num_neurons);
    
y_pred = (sim(net, X_val)>0);
y_pred = 2*y_pred - 1;
    
a = sum(y_pred == y_val);
acc = a/size(y_val,2);

%% section 6.4

X_test = best_feature_test.';

y_test_pred_RBF = (sim(net, X_test)>0);
y_test_pred_RBF = 2*y_test_pred_RBF - 1;

save('labels_test_predicted_RBF.mat', 'y_test_pred_RBF');

%% section 7.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Phase2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%genetic algorithm
%stat

num = 30;

fitnessfcn = @(x) fitness1(train_labels, variance, covariance, FF, num_channel, num_data, num, x);

% Define the lower and upper bounds
lb = ones(1, num);
ub = l_stat * ones(1, num);

% Define the integer constraints
intcon = 1:num;

options = optimoptions('ga', 'MaxGenerations', 200, 'PopulationSize', 100);

[x, fval] = ga(fitnessfcn, num, [], [], [], [], lb, ub, [], intcon, options);

id_best_stat = x;

%% section 7.2
%freq

num = 20;

fitnessfcn = @(x) fitness2(train_labels, f_max, f_mean, f_med, f_ratio1, f_ratio2, f_ratio3, f_ratio4, f_ratio5, f_ratio6, f_ratio7, num_channel, num_data, num, x);

% Define the lower and upper bounds
lb = ones(1, num);
ub = l_freq * ones(1, num);

% Define the integer constraints
intcon = 1:num;

options = optimoptions('ga', 'MaxGenerations', 200, 'PopulationSize', 100);

[x, fval] = ga(fitnessfcn, num, [], [], [], [], lb, ub, [], intcon, options);

id_best_freq = x;

%% section 7.3
%selection

best_feature_genetic = zeros(num_data,50);

variance_selected_genetic = zeros(1,num_channel);
covariance_selected_genetic = zeros(num_channel,num_channel);
FF_selected_genetic = zeros(1,num_channel);

f_max_selected_genetic = zeros(1,num_channel);
f_mean_selected_genetic = zeros(1,num_channel);
f_med_selected_genetic = zeros(1,num_channel);
f_ratio1_selected_genetic = zeros(1,num_channel);
f_ratio2_selected_genetic = zeros(1,num_channel);
f_ratio3_selected_genetic = zeros(1,num_channel);
f_ratio4_selected_genetic = zeros(1,num_channel);
f_ratio5_selected_genetic = zeros(1,num_channel);
f_ratio6_selected_genetic = zeros(1,num_channel);
f_ratio7_selected_genetic = zeros(1,num_channel);

for i=1:30
    l = id_best_stat(i);
    if (l<=num_channel)
        best_feature_genetic(:,i) = variance(:,l);
        variance_selected_genetic(1,l) = 1;
    end
    if l>num_channel && l<= (num_channel + num_channel*(num_channel-1)/2)
        t = l - num_channel;
        for j = 1:(num_channel-1)
            if t<=(num_channel-j) && t>0
                best_feature_genetic(:,i) = covariance(:,j,t + j);
                covariance_selected_genetic(j, t + j) = 1;
                t = 0;
            else
                t = t - (num_channel-j);
            end
        end
    end
    if (l>(num_channel + num_channel*(num_channel-1)/2))
        best_feature_genetic(:,i) = FF(:,(l - (num_channel + num_channel*(num_channel-1)/2)));
        FF_selected_genetic(1,(l - (num_channel + num_channel*(num_channel-1)/2))) = 1;
    end
end

for i=1:20
    l = id_best_freq(i);
    if (l<=num_channel)
        best_feature_genetic(:,(i+30)) = f_max(:,l);
        f_max_selected_genetic(1,l) = 1;
    end
    if l>num_channel && l<=(2*num_channel)
        best_feature_genetic(:,(i+30)) = f_mean(:,l-num_channel);
        f_mean_selected_genetic(1,l-num_channel) = 1;
    end
    if l>2*num_channel && l<=(3*num_channel)
        best_feature_genetic(:,(i+30)) = f_med(:,l-2*num_channel);
        f_med_selected_genetic(1,l-2*num_channel) = 1;
    end
    if l>3*num_channel && l<=(4*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio1(:,l-3*num_channel);
        f_ratio1_selected_genetic(1,l-3*num_channel) = 1;
    end
    if l>4*num_channel && l<=(5*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio2(:,l-4*num_channel);
        f_ratio2_selected_genetic(1,l-4*num_channel) = 1;
    end
    if l>5*num_channel && l<=(6*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio3(:,l-5*num_channel);
        f_ratio3_selected_genetic(1,l-5*num_channel) = 1;
    end
    if l>6*num_channel && l<=(7*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio4(:,l-6*num_channel);
        f_ratio4_selected_genetic(1,l-6*num_channel) = 1;
    end
    if l>7*num_channel && l<=(8*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio5(:,l-7*num_channel);
        f_ratio5_selected_genetic(1,l-7*num_channel) = 1;
    end
    if l>8*num_channel && l<=(9*num_channel)
        best_feature_genetic(:,(i+30)) = f_ratio6(:,l-8*num_channel);
        f_ratio6_selected_genetic(1,l-8*num_channel) = 1;
    end
    if l>9*num_channel
        best_feature_genetic(:,(i+30)) = f_ratio7(:,l-9*num_channel);
        f_ratio7_selected_genetic(1,l-9*num_channel) = 1;
    end
end

%% section 8.1
%MLP

c = cvpartition(num_data,'KFold',k);
net = fitnet([num_layer1 num_layer2]);

train_idx = c.training(1);
test_idx = c.test(1);

X_train = best_feature_genetic(train_idx,:).';
y_train = train_labels(:,train_idx);
X_val = best_feature_genetic(test_idx,:).';
y_val = train_labels(:,test_idx);
    
net = train(net, X_train, y_train);
    
y_pred = (sim(net, X_val)>0);
y_pred = 2*y_pred - 1;
    
a = sum(y_pred == y_val);
acc = a/size(y_val,2);

%% section 8.2

best_feature_test_genetic = zeros(num_data_test,50);

for i=1:30
    l = id_best_stat(i);
    if (l<=num_channel_test)
        best_feature_test_genetic(:,i) = variance_test(:,l);
    end
    if l>num_channel_test && l<= (num_channel_test + num_channel_test*(num_channel_test-1)/2)
        t = l - num_channel_test;
        for j = 1:(num_channel_test-1)
            if t<=(num_channel_test-j) && t>0
                best_feature_test_genetic(:,i) = covariance_test(:,j,t + j);
                t = 0;
            else
                t = t - (num_channel_test-j);
            end
        end
    end
    if (l>(num_channel_test + num_channel_test*(num_channel_test-1)/2))
        best_feature_test_genetic(:,i) = FF_test(:,(l - (num_channel_test + num_channel_test*(num_channel_test-1)/2)));
    end
end

for i=1:20
    l = id_best_freq(i);
    if (l<=num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_max_test(:,l);
    end
    if l>num_channel_test && l<=(2*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_mean_test(:,l-num_channel_test);
    end
    if l>2*num_channel_test && l<=(3*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_med_test(:,l-2*num_channel_test);
    end
    if l>3*num_channel_test && l<=(4*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio1_test(:,l-3*num_channel_test);
    end
    if l>4*num_channel_test && l<=(5*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio2_test(:,l-4*num_channel_test);
    end
    if l>5*num_channel_test && l<=(6*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio3_test(:,l-5*num_channel_test);
    end
    if l>6*num_channel_test && l<=(7*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio4_test(:,l-6*num_channel_test);
    end
    if l>7*num_channel_test && l<=(8*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio5_test(:,l-7*num_channel_test);
    end
    if l>8*num_channel_test && l<=(9*num_channel_test)
        best_feature_test_genetic(:,(i+30)) = f_ratio6_test(:,l-8*num_channel_test);
    end
    if l>9*num_channel_test
        best_feature_test_genetic(:,(i+30)) = f_ratio7_test(:,l-9*num_channel_test);
    end
end

%% section 8.3

X_test_genetic = best_feature_test_genetic.';

y_test_pred_MLP_genetic = (sim(net, X_test_genetic)>0);
y_test_pred_MLP_genetic = 2*y_test_pred_MLP_genetic - 1;

save('labels_test_predicted_MLP_genetic.mat', 'y_test_pred_MLP_genetic');

%% section 9.1
%RBF

c = cvpartition(num_data,'KFold',k);

train_idx = c.training(1);
test_idx = c.test(1);

X_train = best_feature_genetic(train_idx,:).';
y_train = train_labels(:,train_idx);
X_val = best_feature_genetic(test_idx,:).';
y_val = train_labels(:,test_idx);
    
net = newrb(X_train,y_train, 0, Spread, num_neurons);
    
y_pred = (sim(net, X_val)>0);
y_pred = 2*y_pred - 1;
    
a = sum(y_pred == y_val);
acc = a/size(y_val,2);

%% section 9.2

X_test_genetic = best_feature_test_genetic.';

y_test_pred_RBF_genetic = (sim(net, X_test_genetic)>0);
y_test_pred_RBF_genetic = 2*y_test_pred_RBF_genetic - 1;

save('labels_test_predicted_RBF_genetic.mat', 'y_test_pred_RBF_genetic');



