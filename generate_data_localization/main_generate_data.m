clc;
clear

SNR =  10;
len=length(SNR);

Nt = 512;
fc = 6e9;
c = 3e8;
B = 6e9;
lambda_c = c/fc;
d = lambda_c / 2;
M = 2048;
f = zeros(1, M);
for m= 1:M
    %      f(m)=fc+ (m) * B/M;
    %     f(m)=fc+B/(M)*(m-1-(M-1)/2);
    f(m)=fc+B/(2)*(2*m/(M-1)-1);
end

RD = 0.5*Nt*Nt*lambda_c;
FR = 0.62*sqrt(  (Nt * d)^3 /lambda_c    );

N_iter =  5;

K = 1 ;
Rmin = 5;
Rmax = 50;
user_theta_max =    60/180*pi;
user_theta_min =   -60/180*pi;
t0 = clock;


r = zeros(N_iter,1);
theta= zeros(N_iter,1);

y1 = zeros(M,N_iter);
y2 = zeros(M,N_iter);
y3 = zeros(M,N_iter);
y4 = zeros(M,N_iter);
y5 = zeros(M,N_iter);
y6 = zeros(M,N_iter);
y7 = zeros(M,N_iter);
y8 = zeros(M,N_iter);


for i_iter = 1:N_iter

%     i = max(1, round( rand*len));
    random_sequence = randperm(len);
    i = random_sequence(1);
    SNR_linear=10.^(SNR(i)/10.);

    fprintf('  iteration:[%d/%d] |  SNR:[%d/%d]  | run %.4f s\n', i_iter,  N_iter  , i ,  len , etime(clock, t0));


%     r(  i_iter ,:) = rr( random_indices(i_iter)  );
    r(  i_iter ,:) = Rmin + rand * (Rmax - Rmin);

    theta(  i_iter ,:) = user_theta_min + rand * (user_theta_max - user_theta_min);

%     r = 15;
%     theta(  i_iter ,:) = -40/180*pi;

    h= near_field_channel(Nt, d, fc, B, M, r(  i_iter ,:), theta(  i_iter ,:));


        [ r_hat_CBSL  , theta_hat_CBSL ,  y111 , y222 ] = CBS_Low_Localization_Scheme( h,  Nt, M, K,B, d, f,   SNR_linear,   Rmin,    Rmax,   user_theta_max ,  user_theta_min ,theta(  i_iter ,:),r(  i_iter ,:));

        y1(:,i_iter ) = y111(1,:,1) ;
        y2(:,i_iter ) = y111(1,:,2) ;
        y3(:,i_iter ) = y111(1,:,3) ;
        y4(:,i_iter ) = y111(1,:,4) ;
        y5(:,i_iter ) = y222(1,:,1) ;
        y6(:,i_iter ) = y222(1,:,2) ;        


        for m = 1 : M
                y7(m,i_iter) =  theta_hat_CBSL;
        end        

        for m = 1 : M
                y8(m,i_iter) =  r_hat_CBSL;
        end      

end



%%

% y11 = zeros(64,32,N_iter);
% y22 = zeros(64,32,N_iter);
% y33 = zeros(64,32,N_iter);
% y44 = zeros(64,32,N_iter);
% y55 = zeros(64,32,N_iter);
% y66 = zeros(64,32,N_iter);
% y77 = zeros(64,32,N_iter);
% y88 = zeros(64,32,N_iter);
% 
% 
% label = zeros(N_iter,2);
% label(:,1) = r;
% label(:,2) = theta;
% 
% for n = 1 : N_iter
% 
%     y11(:,:,n) = reshape( y1(:,n) , [64,32]  );
%     y22(:,:,n) = reshape( y2(:,n) , [64,32]  );
%     y33(:,:,n) = reshape( y3(:,n) , [64,32]  );
%     y44(:,:,n) = reshape( y4(:,n) , [64,32]  );
%     y55(:,:,n) = reshape( y5(:,n) , [64,32]  );
%     y66(:,:,n) = reshape( y6(:,n) , [64,32]  );
%     y77(:,:,n) = reshape( y7(:,n) , [64,32]  );
%     y88(:,:,n) = reshape( y8(:,n) , [64,32]  );
% 
% 
% end
% 
% x1 = zeros(64,64,N_iter);
% x2 = zeros(64,64,N_iter);
% x3 = zeros(64,64,N_iter);
% x4 = zeros(64,64,N_iter);
% x5 = zeros(64,64,N_iter);
% x6 = zeros(64,64,N_iter);
% x7 = zeros(64,64,N_iter);
% x8 = zeros(64,64,N_iter);
% 
% for n = 1 : N_iter
% 
%     x1(:,1:32,n) =  abs(y11(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y11(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);        
%         x1(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;
%     end
% 
%     x2(:,1:32,n) =  abs(y22(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y22(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);          
%         x2(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;
%     end
% 
%     x3(:,1:32,n) =  abs(y33(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y33(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);        
%         x3(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;        
%     end
% 
%     x4(:,1:32,n) =  abs(y44(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y44(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);        
%         x4(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;            
%     end
% 
%     x5(:,1:32,n) =  abs(y55(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y55(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);        
%         x5(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;                    
%     end
% 
%     x6(:,1:32,n) =  abs(y66(:,:,n));
%     for i = 1 : 32
%         angle_wrapped =  phase(y66(:,i,n));
%         angle_adjusted = unwrap(angle_wrapped);        
%         x6(:,i+32,n) =  mod(angle_adjusted + pi, 2*pi) - pi;                    
%     end
% 
% 
%     x7(:,1:32,n) =  real(y77(:,:,n));
%     x7(:,33:64,n) =  real(y77(:,:,n));    
% 
%     x8(:,1:32,n) =  real(y88(:,:,n));
%     x8(:,33:64,n) =  real(y88(:,:,n));    
% 
% 
% end
