% let us initialize a vector to calculate the power iteration

inp = load('C:\Users\bhada\Downloads\IU_Sem3\MLSP\Assignment\Homework1\Data\flute.mat','X');
%inp_update = inp.X(:,1:128);
inp_update = cov(inp.X.');

imagesc(inp.X);

%inp_update = [4 7 1;0 -3 8;0 0 2];

[V,D,W] = eig(inp_update);

lamb = zeros(2,2);
vec = zeros(128,2);
for i = 1:2
    %if i == 1
    x = ones(128,1);
    [lambda, vect] = powerMethod(inp_update,x);
    lamb(i,i) = lambda;
    vec(:,i) = vect;
    [B] = createMatrix(lamb(i),vec(:,i),inp_update);
    inp_update = B;
end

% Plotting the two basis vector
imagesc(vec);

act
% calculation of an activation function

act = zeros(2,128);

act = lamb * vec.';
act1 = vec' * inp.X;
imagesc(act1);

v1 = vec(:,1)' * inp.X;
imagesc(v1)
    
function [B] = createMatrix(lambda,vect,inp_update)
    check_transpose = vect * vect.';
    mod_value = check_transpose./power(norm(vect),2);
    mid_result = lambda.*mod_value;
    B = inp_update - mid_result;
end

function [lambda_value,x] = powerMethod(inp_update,x)
for i= 1:1000
    m = inp_update*x;
    [V,I] = max(abs(m));
    lambda_value = m(I);
    x = m./lambda_value;
end
end




