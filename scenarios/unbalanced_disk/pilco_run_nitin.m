%% 1. Initialization
clear all; close all; clc
settings_unbalanced_disk;            % load scenario-specific settings
basename = 'disk_';       % filename used for saving data

% 2. Initial J random rollouts
for jj = 1:J
  [xx, yy, realCost{jj}, latent{jj}] = ...
    rollout(gaussian(mu0, S0), struct('maxU',policy.maxU), H, plant, cost);
  x = [x; xx]; y = [y; yy];       % augment training sets for dynamics model
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_pendulum;
  end
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

% 3. Controlled learning (N iterations)
for j = 1:N
  trainDynModel;   % train (GP) dynamics model
  learnPolicy;     % learn policy
  applyController; % apply controller to system
  disp(['controlled trial # ' num2str(j)]);
  if plotting.verbosity > 0;      % visualization of trajectory
    if ~ishandle(1); figure(1); else set(0,'CurrentFigure',1); end; clf(1);
    draw_rollout_pendulum;
  end
end


%state = init_state_disk();
%dt = 0.025;
%
%%% Dynamic run
%for i = 1:101
%    thetas(i) = state.theta;
%    u = 3;
%    state = unbalanced_disk(state, u, dt);
%end
%
%%% Plotting
%plot(thetas)
%title(num2str(max(thetas)))
