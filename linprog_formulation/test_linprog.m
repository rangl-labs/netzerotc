%%

%%
% load('rewards_all.mat')
load('rewards_all_reward_plus_jobs_increment.mat')
f = -rewards;
A = [capex_all, -jobs_1Yincrements, -jobs_2Yincrements].';
% A = [capex_all, -jobs_1Yincrements(:,1:end-1), -jobs_2Yincrements(:,1:end-1)].';
b = [repelem(26390,size(capex_all,2)) repelem(25000,size(jobs_1Yincrements,2)) repelem(37500,size(jobs_2Yincrements,2))].';
% b = [repelem(26390,size(capex_all,2)) repelem(25000,size(jobs_1Yincrements(:,1:end-1),2)) repelem(37500,size(jobs_2Yincrements(:,1:end-1),2))].';
lb = zeros(size(f));
ub = [repelem(150,20) repelem(270,20) repelem(252.797394,20)].';
Aeq = [];
beq = [];
[x,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq,lb,ub);

actions = reshape(x,20,3);

max_reward = -fval;
