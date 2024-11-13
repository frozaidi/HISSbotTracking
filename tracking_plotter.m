listing = dir("processed_csv/HG_rev*.csv");
table = struct2table(listing);
filenames = strcat(table.folder,"\",table.name);

color = {[1 0 0], [0 1 0], [0 0 1]};

f = figure(1);
axis equal
hold on
for i = 1:numel(filenames)
    T = readtable(filenames{i});
    x = T.X;
    y = T.Y;
    
    [x_proc, y_proc] = process_data(x,y,10);
    plot(x_proc,y_proc)
    dist = sqrt(x_proc(end)^2+y_proc(end)^2)/10/0.9;
    fprintf('x = %.3f; y = %.3f\n',x_proc(end),y_proc(end));
    fprintf('BL/Cycle = %.2f\n', dist);
end