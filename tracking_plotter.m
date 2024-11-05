listing = dir("*.csv");
table = struct2table(listing);
filenames = table.name;

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

end