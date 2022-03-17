function FaceRecognition
%本算法使用的是orl_faces数据集(ORL人脸数据集共包含40个不同人的400张图像,是在1992年4月至1994年4月期间由英国剑桥的Olivetti研究实验室创建)
clear  % calc xmean,sigma and its eigen decomposition  
close all
%%
allsamples=[];%所有训练图像 
syms line %line用来分割10组（一共有40个人，每个人有10张照片，40个人的一张照片作为一组）照片，分别用来训练和测试准确度
line=9;%用一个人的9张照片训练，剩下一张用来测试
for i=1:40%40个人    
    for j=1:line %取每个人的前line张照片       
           a=imread(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j),'.pgm'));          
        b=a(1:112*92); % b是行矢量 1×N，其中N＝10304，提取顺序是先列后行，即从上到下，从左到右        
        b=double(b);        
        allsamples=[allsamples; b];  % allsamples 是一个M * N 矩阵，allsamples 中每一行数据代表一张图片，其中M＝200   
    end
end
%%
train=40*line;%用来训练的总图片数量
samplemean=mean(allsamples); % 平均图片，1 × N  
figure%平均图
display('平均脸：')
imshow(mat2gray(reshape(samplemean,112,92)));
%%
for i=1:train 
    xmean(i,:)=allsamples(i,:)-samplemean; % xmean是一个M × N矩阵，xmean每一行保存的数据是“每个图片数据-平均图片” 
end;   
figure%差值图
display('差值脸：')
imshow(mat2gray(reshape(xmean(1,:),112,92)));
%%
sigma=xmean*xmean';   % M * M 阶矩阵 
[v,d]=eig(sigma);
d1=diag(d); 
[d2,index]=sort(d1); %以升序排序 
cols=size(v,2);% 特征向量矩阵的列数

for i=1:cols      
    vsort(:,i) = v(:, index(cols-i+1) ); % vsort 是一个M*col(注:col一般等于M)阶矩阵，保存的是按降序排列的特征向量,每一列构成一个特征向量      
    dsort(i)   = d1( index(cols-i+1) );  % dsort 保存的是按降序排列的特征值，是一维行向量 
end  %完成降序排列 %以下选择90%的能量 
dsum = sum(dsort);     
dsum_extract = 0;   
p = 0;     
while( dsum_extract/dsum < 0.9)       
    p = p + 1;          
    dsum_extract = sum(dsort(1:p));     
end
a=1:1:train;
for i=1:1:train
y(i)=sum(dsort(a(1:i)) );
end
%%
figure
y1=ones(1,train);
plot(a,y/dsum,a,y1*0.9,'linewidth',2);
grid
title('前n个特征特占总的能量百分比');
xlabel('前n个特征值');
ylabel('占百分比');
%%
figure
plot(a,dsort/dsum,'linewidth',2);
grid
title('第n个特征特占总的能量百分比');
xlabel('第n个特征值');
ylabel('占百分比');
%%
i=1;  % (训练阶段)计算特征脸形成的坐标系
while (i<=p && dsort(i)>0)      
    base(:,i) = dsort(i)^(-1/2) * xmean' * vsort(:,i);   % base是N×p阶矩阵，除以dsort(i)^(1/2)是对人脸图像的标准化，特征脸
      i = i + 1; 
end
%%
allcoor = allsamples * base; accu = 0;   % 测试过程
for i=1:40     
     for j=(line+1):10 %读入数据集中剩下的测试图像       
        a=imread(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j),'.pgm'));     
        b=a(1:10304);        
        b=double(b);        
        tcoor= b * base; %计算坐标，是1×p阶矩阵      
        for k=1:(train)                 
            mdist(k)=norm(tcoor-allcoor(k,:));        
        end;          %三阶近邻   
        [dist,index2]=sort(mdist);              
        class1=floor(index2(1)/line)+1;      
        class2=floor(index2(2)/line)+1;        
        class3=floor(index2(3)/line)+1;        
        if class1~=class2 && class2~=class3 
            class=class1;         
        elseif class1==class2          
            class=class1;         
        elseif class2==class3     
            class=class2;         
        end;         
        if class==i      
            accu=accu+1;        
        end;   
    end;
end;  
display('模型识别率')
accuracy=accu/(400-train) %输出识别率
%%
display('比如说，我想找这第七个人，这是他第10张照片')
i1=7; j1=10;%数据集中的第7个人，其第10张照片
figure
imshow((strcat('E:\orl_faces\s',num2str(i1),'\',num2str(j1),'.pgm')));
for i=1:40     
        a=imread(strcat('E:\orl_faces\s',num2str(i1),'\',num2str(j1),'.pgm'));%这行意思是告诉机器要找谁
        b=a(1:10304);        
        b=double(b);        
        tcoor= b * base; %计算坐标，是1×p阶矩阵      
        for k=1:train
            mdist(k)=norm(tcoor-allcoor(k,:));        
        end;
        [dist,index2]=sort(mdist); %三阶近邻              
        class1=floor(index2(1)/line)+1;      
        class2=floor(index2(2)/line)+1;        
        class3=floor(index2(3)/line)+1;        
        if class1~=class2 && class2~=class3 
            class=class1;         
        elseif class1==class2          
            class=class1;         
        elseif class2==class3     
            class=class2;         
        end;         
        if class==i      
            figure%平均图
            display('选了s7的第10张照片，该张照片没训练过，机器不认识，如果识别出这张照片是s7这个人，那就随便挑一张（这里就挑j1-1即第9张）显示')
            imshow(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j1-1),'.pgm'));    
        end;   
end
