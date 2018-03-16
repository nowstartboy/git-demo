function new_D = get_multiFeature(D,d)
[n,m] = size(D); 
new_D = ones(n,1);
x1=D(:,1);
x2=D(:,2);
for k = 1:d
    for l = 0:k
        new_D(:, end+1) = (x1.^(k-l)).*(x2.^l);
    end
end


end

