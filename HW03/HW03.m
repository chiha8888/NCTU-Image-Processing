clc;
clear all;

% original scale
I = rgb2gray(imread('1.jpg'));
% x0.5 scale
%I = imresize(I, 0.5);
% x0.25 scale
%I = imresize(I, 0.25);
block_size=4;
matrix =2;
method = 1;

[row, coln] = size(I);
I1 = I;
I = double(I);
restored = zeros(row,coln);
% Hadamard Matrix
H = hadamard(block_size);
iH = H';
% DCT Matrix
DCT = dctmtx(block_size);
iDCT = DCT';
% DFT Matrix
DFT = dftmtx(block_size);
iDFT = DFT';


% Forward WHT, DCT, DFT
for i1 = [1:block_size:row]
    for i2 = [1:block_size:coln]
        imBlock = I(i1:i1+block_size-1,i2:i2+block_size-1);
        
        % WHT
        if(matrix == 1)
            win1 = H * imBlock * iH;
        end
        % DCT
        if(matrix == 2)
            win1 = DCT * imBlock * iDCT;
        end
        % DFT
        if(matrix == 3)
            win1 = DFT * imBlock * iDFT;
        end
        domain(i1:i1+block_size-1,i2:i2+block_size-1) = win1;
    end
end

% Variance matrix
mean = zeros(block_size,block_size);
var = zeros(block_size,block_size);
c = 0;
for i1 = [1:block_size:row]
    for i2 = [1:block_size:coln]
        win1 = domain(i1:i1+block_size-1,i2:i2+block_size-1);
        mean = mean + win1;
        c = c + 1;
    end
end
mean = mean / c;
for i1 = [1:block_size:row]
    for i2 = [1:block_size:coln]
        win1 = domain(i1:i1+block_size-1,i2:i2+block_size-1);
        tmp = win1 - mean;
        for j1 = 1:block_size
            for j2 = 1:block_size
                tmp(j1,j2) = tmp(j1,j2)^2;
            end
        end
        var = var + tmp;
    end
end
var = var / c;
q = log(var);


% Quantization
for i1 = [1:block_size:row]
    for i2 = [1:block_size:coln]
        win1 = domain(i1:i1+block_size-1,i2:i2+block_size-1);
        
        % Keep only the first 9 coefficients
        if(method == 1)
            for j1 = 1:block_size
                for j2 = 1:block_size
                    if(j1 < 4 && j2 < 4)
                        win2(j1,j2) = win1(j1,j2);
                    else
                        win2(j1,j2) = 0;
                    end
                end
            end
        end
        
        % Keep only the coefficients with the k largest coefficients
        if(method == 2)
            win2 = zeros(block_size,block_size);
            for j1 = 1:9
                [x,y] = find(win1 == max(win1(:)));
                win2(x,y) = win1(x,y); 
                win1(x,y) = -100;
            end
        end
        
        % logarithm of coefficient variances with 20bits
        if(method == 3)
            sumq = 0;
            for j1 = 1:block_size
                for j2 = 1:block_size
                    sumq = sumq + q(j1,j2);
                end
            end
            n = zeros(block_size,block_size);
            for j1 = 1:block_size
                for j2 = 1:block_size
                    n(j1,j2) = 20 * q(j1,j2) / sumq;
                    win2(j1,j2) = win1(j1,j2) * n(j1,j2);
                end
            end
        end
        
        quantized(i1:i1+block_size-1,i2:i2+block_size-1) = win2;
    end
end

% Inverse WHT, DCT, DFT
for i1 = [1:block_size:row]
    for i2 = [1:block_size:coln]
        win3 = quantized(i1:i1+block_size-1,i2:i2+block_size-1);
        
        % WHT
        if(matrix == 1)
            win4 = iH * win3 * H;
        end
        % DCT
        if(matrix == 2)
            win4 = iDCT * win3 * DCT;
        end
        % DFT
        if(matrix == 3)
            win4 = iDFT * win3 * DFT;
        end
        restored(i1:i1+block_size-1,i2:i2+block_size-1) = win4;
    end
end

% RMSE, SNR
add = 0;
SNR = 0;
for i1 = 1:row
    for i2 = 1:coln
        temp = (I(i1,i2) - restored(i1,i2)) ^ 2;
        add = add + temp;
        
        temp = I(i1,i2) ^ 2;
        SNR = SNR + temp;
    end
end
RMSE = (add / (row * coln)) ^ 0.5;
SNR =  10 * log10(SNR / add);
fprintf('RMSE: %.2f \n',RMSE);
fprintf('SNR: %.2f',SNR);
I2 = restored;
% I2 = mat2gray(I2);
% for DFT
I2 = mat2gray(abs(I2));

% Display
% figure(1);imshow(I1);title('original image');
figure(2);imshow(I2);