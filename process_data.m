function [x_processed, y_processed] = process_data(x,y,window)
    
    x = x - x(1);
    y = y - y(1);
    
    
    [y_outlier, TFrm] = rmoutliers(y);
    x_outlier = x(~TFrm);

    [x_outlier,TFrm] = rmoutliers(x_outlier);
    y_outlier = y_outlier(~TFrm);
    
    
    x_processed = movmean(x_outlier,window);
    y_processed = movmean(y_outlier,window);
end