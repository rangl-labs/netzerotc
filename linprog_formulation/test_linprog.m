%%

%%
f = -rewards;
A = eye(60);
b = [repelem(150,20) repelem(270,20) repelem(252.797394,20)].';
lb = zeros(size(f));
ub = b;
Aeq = [];
beq = [];
[x,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq,lb,ub);

actions = reshape(x,20,3);

max_reward = -fval;
