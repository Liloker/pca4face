function FaceRecognition
%���㷨ʹ�õ���orl_faces���ݼ�(ORL�������ݼ�������40����ͬ�˵�400��ͼ��,����1992��4����1994��4���ڼ���Ӣ�����ŵ�Olivetti�о�ʵ���Ҵ���)
clear  % calc xmean,sigma and its eigen decomposition  
close all
%%
allsamples=[];%����ѵ��ͼ�� 
syms line %line�����ָ�10�飨һ����40���ˣ�ÿ������10����Ƭ��40���˵�һ����Ƭ��Ϊһ�飩��Ƭ���ֱ�����ѵ���Ͳ���׼ȷ��
line=9;%��һ���˵�9����Ƭѵ����ʣ��һ����������
for i=1:40%40����    
    for j=1:line %ȡÿ���˵�ǰline����Ƭ       
           a=imread(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j),'.pgm'));          
        b=a(1:112*92); % b����ʸ�� 1��N������N��10304����ȡ˳�������к��У������ϵ��£�������        
        b=double(b);        
        allsamples=[allsamples; b];  % allsamples ��һ��M * N ����allsamples ��ÿһ�����ݴ���һ��ͼƬ������M��200   
    end
end
%%
train=40*line;%����ѵ������ͼƬ����
samplemean=mean(allsamples); % ƽ��ͼƬ��1 �� N  
figure%ƽ��ͼ
display('ƽ������')
imshow(mat2gray(reshape(samplemean,112,92)));
%%
for i=1:train 
    xmean(i,:)=allsamples(i,:)-samplemean; % xmean��һ��M �� N����xmeanÿһ�б���������ǡ�ÿ��ͼƬ����-ƽ��ͼƬ�� 
end;   
figure%��ֵͼ
display('��ֵ����')
imshow(mat2gray(reshape(xmean(1,:),112,92)));
%%
sigma=xmean*xmean';   % M * M �׾��� 
[v,d]=eig(sigma);
d1=diag(d); 
[d2,index]=sort(d1); %���������� 
cols=size(v,2);% �����������������

for i=1:cols      
    vsort(:,i) = v(:, index(cols-i+1) ); % vsort ��һ��M*col(ע:colһ�����M)�׾��󣬱�����ǰ��������е���������,ÿһ�й���һ����������      
    dsort(i)   = d1( index(cols-i+1) );  % dsort ������ǰ��������е�����ֵ����һά������ 
end  %��ɽ������� %����ѡ��90%������ 
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
title('ǰn��������ռ�ܵ������ٷֱ�');
xlabel('ǰn������ֵ');
ylabel('ռ�ٷֱ�');
%%
figure
plot(a,dsort/dsum,'linewidth',2);
grid
title('��n��������ռ�ܵ������ٷֱ�');
xlabel('��n������ֵ');
ylabel('ռ�ٷֱ�');
%%
i=1;  % (ѵ���׶�)�����������γɵ�����ϵ
while (i<=p && dsort(i)>0)      
    base(:,i) = dsort(i)^(-1/2) * xmean' * vsort(:,i);   % base��N��p�׾��󣬳���dsort(i)^(1/2)�Ƕ�����ͼ��ı�׼����������
      i = i + 1; 
end
%%
allcoor = allsamples * base; accu = 0;   % ���Թ���
for i=1:40     
     for j=(line+1):10 %�������ݼ���ʣ�µĲ���ͼ��       
        a=imread(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j),'.pgm'));     
        b=a(1:10304);        
        b=double(b);        
        tcoor= b * base; %�������꣬��1��p�׾���      
        for k=1:(train)                 
            mdist(k)=norm(tcoor-allcoor(k,:));        
        end;          %���׽���   
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
display('ģ��ʶ����')
accuracy=accu/(400-train) %���ʶ����
%%
display('����˵������������߸��ˣ���������10����Ƭ')
i1=7; j1=10;%���ݼ��еĵ�7���ˣ����10����Ƭ
figure
imshow((strcat('E:\orl_faces\s',num2str(i1),'\',num2str(j1),'.pgm')));
for i=1:40     
        a=imread(strcat('E:\orl_faces\s',num2str(i1),'\',num2str(j1),'.pgm'));%������˼�Ǹ��߻���Ҫ��˭
        b=a(1:10304);        
        b=double(b);        
        tcoor= b * base; %�������꣬��1��p�׾���      
        for k=1:train
            mdist(k)=norm(tcoor-allcoor(k,:));        
        end;
        [dist,index2]=sort(mdist); %���׽���              
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
            figure%ƽ��ͼ
            display('ѡ��s7�ĵ�10����Ƭ��������Ƭûѵ��������������ʶ�����ʶ���������Ƭ��s7����ˣ��Ǿ������һ�ţ��������j1-1����9�ţ���ʾ')
            imshow(strcat('E:\orl_faces\s',num2str(i),'\',num2str(j1-1),'.pgm'));    
        end;   
end
