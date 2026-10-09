% Example for ORF
clear all;
T=0:0.01:10;% Time construction

% Set of slopes
A=[2 -1 4 -3 -1];

%Knots
Theta1=[0,2,4,6,8]; % Equidistant
Theta2=[0,3,5,6,7]; %Non-equididtant

%Intercept
b=0;

% Equidistant spline S1, S2-nonequidistant
% create spline: S1-equidistant, S2- non-equidistant

n=length(T);

 for i=1:n
     S1(i)=b;
     S2(i)=b;
     for j=1:5
 S1(i)=S1(i)+A(j)*max(0,T(i)-Theta1(j));
 S2(i)=S2(i)+A(j)*max(0,T(i)-Theta2(j));
 end;
end;

plot(T,S1);
hold on
plot(T,S2);

dlmwrite("S1.txt", S1);
dlmwrite("S2.txt", S2);
