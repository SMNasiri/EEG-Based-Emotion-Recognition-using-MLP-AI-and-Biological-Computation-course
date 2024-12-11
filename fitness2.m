function score = fitness2(train_labels, f_max, f_mean, f_med, f_ratio1, f_ratio2, f_ratio3, f_ratio4, f_ratio5, f_ratio6, f_ratio7, num_channel, num_data, num, x)
    mu_0 = zeros(num,1);
    mu_1 = zeros(num,1);
    mu_2 = zeros(num,1);
    best_feature_gen = zeros(num_data,num);
    labels = train_labels.';
    for i=1:num
        l = x(i);
        if (l<=num_channel)
            best_feature_gen(:,i) = f_max(:,l);
            mu_0(i,1) = mean(f_max(:,l));
            mu_1(i,1) = mean(f_max((labels == 1),l));
            mu_2(i,1) = mean(f_max((labels == -1),l));
        end
        if l>num_channel && l<=(2*num_channel)
            best_feature_gen(:,i) = f_mean(:,l-num_channel);
            mu_0(i,1) = mean(f_mean(:,l-num_channel));
            mu_1(i,1) = mean(f_mean((labels == 1),l-num_channel));
            mu_2(i,1) = mean(f_mean((labels == -1),l-num_channel));
        end
        if l>2*num_channel && l<=(3*num_channel)
            best_feature_gen(:,i) = f_med(:,l-2*num_channel);
            mu_0(i,1) = mean(f_med(:,l-2*num_channel));
            mu_1(i,1) = mean(f_med((labels == 1),l-2*num_channel));
            mu_2(i,1) = mean(f_med((labels == -1),l-2*num_channel));
        end
        if l>3*num_channel && l<=(4*num_channel)
            best_feature_gen(:,i) = f_ratio1(:,l-3*num_channel);
            mu_0(i,1) = mean(f_ratio1(:,l-3*num_channel));
            mu_1(i,1) = mean(f_ratio1((labels == 1),l-3*num_channel));
            mu_2(i,1) = mean(f_ratio1((labels == -1),l-3*num_channel));
        end
        if l>4*num_channel && l<=(5*num_channel)
            best_feature_gen(:,i) = f_ratio2(:,l-4*num_channel);
            mu_0(i,1) = mean(f_ratio2(:,l-4*num_channel));
            mu_1(i,1) = mean(f_ratio2((labels == 1),l-4*num_channel));
            mu_2(i,1) = mean(f_ratio2((labels == -1),l-4*num_channel));
        end
        if l>5*num_channel && l<=(6*num_channel)
            best_feature_gen(:,i) = f_ratio3(:,l-5*num_channel);
            mu_0(i,1) = mean(f_ratio3(:,l-5*num_channel));
            mu_1(i,1) = mean(f_ratio3((labels == 1),l-5*num_channel));
            mu_2(i,1) = mean(f_ratio3((labels == -1),l-5*num_channel));
        end
        if l>6*num_channel && l<=(7*num_channel)
            best_feature_gen(:,i) = f_ratio4(:,l-6*num_channel);
            mu_0(i,1) = mean(f_ratio4(:,l-6*num_channel));
            mu_1(i,1) = mean(f_ratio4((labels == 1),l-6*num_channel));
            mu_2(i,1) = mean(f_ratio4((labels == -1),l-6*num_channel));
        end
        if l>7*num_channel && l<=(8*num_channel)
            best_feature_gen(:,i) = f_ratio5(:,l-7*num_channel);
            mu_0(i,1) = mean(f_ratio5(:,l-7*num_channel));
            mu_1(i,1) = mean(f_ratio5((labels == 1),l-7*num_channel));
            mu_2(i,1) = mean(f_ratio5((labels == -1),l-7*num_channel));
        end
        if l>8*num_channel && l<=(9*num_channel)
            best_feature_gen(:,i) = f_ratio6(:,l-8*num_channel);
            mu_0(i,1) = mean(f_ratio6(:,l-8*num_channel));
            mu_1(i,1) = mean(f_ratio6((labels == 1),l-8*num_channel));
            mu_2(i,1) = mean(f_ratio6((labels == -1),l-8*num_channel));
        end
        if l>9*num_channel
            best_feature_gen(:,i) = f_ratio7(:,l-9*num_channel);
            mu_0(i,1) = mean(f_ratio7(:,l-9*num_channel));
            mu_1(i,1) = mean(f_ratio7((labels == 1),l-9*num_channel));
            mu_2(i,1) = mean(f_ratio7((labels == -1),l-9*num_channel));
        end
    end
    
    s1 = zeros(num,num);
    s2 = zeros(num,num);
    N1 = 0;
    N2 = 0;
    
    for i=1:num_data
        if (train_labels(1,i)==1)
            s1 = s1 + (best_feature_gen(i,:).' - mu_1)*(best_feature_gen(i,:).' - mu_1).';
            N1 = N1 + 1;
        end
        if (train_labels(1,i)==-1)
            s2 = s2 + (best_feature_gen(i,:).' - mu_2)*(best_feature_gen(i,:).' - mu_2).';
            N2 = N2 + 1;
        end
    end
    s1 = s1/N1;
    s2 = s2/N2;
    
    sw = s1 + s2;
    sb = (mu_1 - mu_0)*(mu_1 - mu_0).' + (mu_2 - mu_0)*(mu_2 - mu_0).';
    
    n = 0;
    for i =1:num
        for j = (i+1):num
            if (x(i)==x(j))
                n = n+1;
            end
        end
    end
    
    score = n -(trace(sb)/trace(sw));
end