% ball=double(ball);
% %size 415x422x3
% smallball=zeros(42,42);
%     for j=1:41
%         smallball(i,j)=reshape(mean(mean(mean(ball(10*(i-1)+1:10*i, 10*(j-1)+1:10*j, 1:3)))),1,1);
%     end
% end
% for j=1:41
% for i=1:41
% smallball(42,j)=reshape(mean(mean(mean(ball(411:415,10*(j-1)+1:10*j,1:3)))),1,1);
% smallball(j, 42)=reshape(mean(mean(mean(ball(10*(j-1)+1:10*j,411:422,1:3)))),1,1);
% end
% smallball(42, 42)=reshape(mean(mean(mean(ball(411:415,411:422,1:3)))),1,1);
clear 
close all
newData1 = load('-mat', 'C:\Users\Beate\Desktop\Hierarchies\smallball42x42.mat');

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
for i = 1:length(vars)
    assignin('base', vars{i}, newData1.(vars{i}));
end
imagesc(smallball)
% [Gmag,Gdir] = imgradient(smallball);
% A=-Gmag; 
% imagesc(Gmag)

[Gx,Gy] = imgradientxy(smallball);
Gx=abs(Gx); Gy=abs(Gy);
Gx(1,:)=[];
Gy(:,1)=[];
%42*41*2=3444
s=zeros(1,3444);
t=zeros(1,3444);
c=1;
%neighbors=zeros(4,3444);
for j=1:42
    for i=1:41  
    s(c)=42*(j-1)+i;%does not define s(42), s(84)...
    t(c)=42*(j-1)+i+1;
    c=c+1;
    end
end
c=1;
for i=1:41
    for j=1:42
    s(1722+c)=42*(i-1)+j;
    t(1722+c)=42*i+j;
    c=c+1;
    end
end
%41*42=1722
weight=zeros(1,3444);

for i=1:1722
    weight(i)=Gx(i);   
    weight(i+1722)=Gy(i);
end
weight=abs(weight); %remove signs of differences.
%x = [0 0.5 -0.5 -0.5 0.5 0 1.5 0 2 -1.5 -2];
%y = [0 0.5 0.5 -0.5 -0.5 2 0 -2 0 0 0];
%z = [5 3 3 3 3 0 1 0 0 1 0];
%plot(G,'XData',x,'YData',y,'ZData',z,'EdgeLabel',G.Edges.Weight)
x= repmat(1: 42, 1,42);
y=[];
for i=1:42
y=[y,(43-i).*ones(1,42)];
end
z=zeros(1,1764);
G=graph(s,t,weight);
figure
pathedges=find(abs(G.Edges.Weight)>20);
p=plot(G,'XData',x,'YData',y,'ZData',z, 'EdgeLabel',floor(G.Edges.Weight));
highlight(p,'Edges',pathedges,'EdgeColor','r', 'LineWidth',1)

[T,pred] = minspantree(G,'Type','forest');
highlight(p,T, 'LineWidth',3)
xlim([-1,43])
ylim([-1,43])
view(2)

%For calculating the minmax weights, use 
%neighbors(G,100)
%find(G.Edges.EndNodes(1)==5)
%A=G.Edges.EndNodes;
A=[G.Edges.EndNodes, G.Edges.Weight];
Astar=zeros(3444,1);
for i=1:3444
    te=neighbors(G,A(i,1));
    te(find(te==A(i,2)))=[];
    m1=min(A(te,3));
    te=neighbors(G,A(i,2));
    te(find(te==A(i,1)))=[];
    Astar(i)=max(min(A(te,3)),m1);
end
 wh=find(Astar<A(:,3)-5) ;
 
 
figure
G2=graph(string(s),string(t),weight);
p=plot(G2,'XData',x,'YData',y,'ZData',z, 'EdgeLabel',floor(G2.Edges.Weight));
highlight(p,'Edges',wh,'EdgeColor','g', 'LineWidth',3)
xlim([-1,43])
ylim([-1,43])

figure
for i=1:size(wh,1)
     v=sort(A(wh(i),1:2));
G2 = rmedge(G2, num2str(A(wh(i),1)),num2str(A(wh(i),2)));
end
bins=conncomp(G2);
k=plot(G2,'XData',x,'YData',y,'ZData',z, 'EdgeLabel',floor(G2.Edges.Weight));
hold on
c=['r';'g';'b';'y';'c'; 'm'; 'k'; 'w'];
w=[  630 ;  524  ; 602  ; 549 ;  194  ; 281;     1];
for i=1:7
    highlight(k, find(bins==w(i)));
highlight(k, find(bins==w(i)),'NodeColor', c(i));

end
xlim([-1,43])
ylim([-1,43])
view(2)

p=plot(G2,'XData',x,'YData',y,'ZData',z, 'EdgeLabel',floor(G2.Edges.Weight));
highlight(p,'Edges',wh,'EdgeColor','g', 'LineWidth',3)
xlim([-1,43])
ylim([-1,43])


%top right (left?) corner
%idx = [30*42+30:31*42, 31*42+30:32*42, 32*42+30:33*42 , 33*42+30:34*42, 34*42+30:35*42, 35*42+30:36*42, ...
%    36*42+30:37*42, 37*42+30:38*42,  38*42+30:39*42, 39*42+30:40*42, 40*42+30:41*42, 41*42+30:42*42];
%bottom left corner (rotated 90 degrees): 
% idx = [30:1*42, 1*42+30:2*42, 2*42+30:3*42 , 3*42+30:4*42, 4*42+30:5*42, 5*42+30:6*42, ...
%    6*42+30:7*42, 7*42+30:8*42,  8*42+30:9*42, 9*42+30:10*42, 10*42+30:11*42, 11*42+30:12*42,...
%    12*42+30:13*42 , 13*42+30:14*42, 14*42+30:15*42, 15*42+30:16*42, 16*42+30:17*42, 17*42+30:18*42,  18*42+30:19*42, 19*42+30:20*42, 20*42+30:21*42];
% H = subgraph(G,idx);
% pathed=find(abs(H.Edges.Weight)<250);
% q=plot(H,'EdgeLabel',floor(H.Edges.Weight));
% highlight(q,'Edges',pathed ,'EdgeColor','r', 'LineWidth',2) 