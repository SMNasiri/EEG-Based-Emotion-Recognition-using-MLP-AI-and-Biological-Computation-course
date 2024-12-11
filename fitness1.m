function score = fitness1(train_labels, variance, covariance, FF, num_channel, num_data, num, x)
    mu_0 = zeros(num,1);
    mu_1 = zeros(num,1);
    mu_2 = zeros(num,1);
    best_feature_gen = zeros(num_data,num);
    labels = train_labels.';
    for i=1:num
        l = x(i);
        if (l<=num_channel)
            best_feature_gen(:,i) = variance(:,l);
            mu_0(i,1) = mean(variance(:,l));
            mu_1(i,1) = mean(variance((labels == 1),l));
            mu_2(i,1) = mean(variance((labels == -1),l));
        end
        if l>num_channel && l<= (num_channel + num_channel*(num_channel-1)/2)
            t = l - num_channel;
            for j = 1:(num_channel-1)
                if t<=(num_channel-j) && t>0
                    best_feature_gen(:,i) = covariance(:,j,t + j);
                    mu_0(i,1) = mean(covariance(:,j,t + j));
                    mu_1(i,1) = mean(covariance((labels == 1),j,t + j));
                    mu_2(i,1) = mean(covariance((labels == -1),j,t + j));
                    t = 0;
                else
                    t = t - (num_channel-j);
                end
            end
        end
        if (l>(num_channel + num_channel*(num_channel-1)/2))
            best_feature_gen(:,i) = FF(:,(l - (num_channel + num_channel*(num_channel-1)/2)));
            mu_0(i,1) = mean(FF(:,(l - (num_channel + num_channel*(num_channel-1)/2))));
            mu_1(i,1) = mean(FF((labels == 1),(l - (num_channel + num_channel*(num_channel-1)/2))));
            mu_2(i,1) = mean(FF((labels == -1),(l - (num_channel + num_channel*(num_channel-1)/2))));
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