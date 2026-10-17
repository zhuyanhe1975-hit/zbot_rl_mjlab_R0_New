"""Exercise real success-triggered command alternation without PPO updates."""
import json
from dataclasses import asdict
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import zbot_rl_mjlab
from zbot_rl_mjlab.tasks.transformation import commanded,mdp,robust
cfg=load_env_cfg(commanded.TASK_ID,play=True)
cfg.scene.num_envs=8
cfg.seed=42
env=ManagerBasedRlEnv(cfg,device='cuda:0')
report={'environments':8,'steps':900,'simulation_seconds':18,'training_updates':0,'switches':[]}
try:
    agent=load_rl_cfg(commanded.TASK_ID)
    wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
    runner=robust.RobustSnakeToBipedRunner(wrapped,asdict(agent),device=env.device)
    runner.load(str(Path(__file__).resolve().parent/'model_599.pt'),load_cfg={'actor':True},map_location=env.device)
    policy=runner.get_inference_policy(device=env.device)
    term=env.command_manager.get_term('transform')
    data=env.scene['robot'].data
    action=env.action_manager.get_term('joint_pos')
    reward_command=term.command.clone()
    metric_command=term.command.clone()
    reward_compute=env.reward_manager.compute
    metric_compute=env.metrics_manager.compute
    update=term._update_command
    def audited_reward(*args,**kwargs):
        reward_command.copy_(term.command)
        return reward_compute(*args,**kwargs)
    def audited_metrics(*args,**kwargs):
        metric_command.copy_(term.command)
        return metric_compute(*args,**kwargs)
    def audited_update(ids):
        before=term.command.clone()
        held=term.switch_hold_time.clone()
        tensors=[data.root_link_pose_w,data.root_link_vel_w,data.joint_pos,data.joint_vel,action.commanded_target,env.episode_length_buf]
        saved=[v.clone() for v in tensors]
        update(ids)
        changed=(term.command[:,0]!=before[:,0]).nonzero().flatten()
        if len(changed):
            assert torch.equal(before[changed],reward_command[changed])
            assert torch.equal(before[changed],metric_command[changed])
            for current,old in zip(tensors,saved,strict=True):assert torch.equal(current,old)
            assert mdp.state(env).hold_time[changed].eq(0).all()
            assert term.switch_hold_time[changed].eq(0).all()
            assert not term.manual_control[changed].any()
            for i in changed.tolist():
                report['switches'].append({'env':i,'step':env.common_step_counter,'from':int(before[i,0]),'to':int(term.command[i,0]),'stable_hold_s':float(held[i]),'episode_step':int(env.episode_length_buf[i])})
    env.reward_manager.compute=audited_reward
    env.metrics_manager.compute=audited_metrics
    term._update_command=audited_update
    obs=wrapped.get_observations()
    done_count=0
    reset_steps=[]
    with torch.inference_mode():
        for _ in range(900):
            obs,reward,done,extra=wrapped.step(policy(obs))
            assert torch.isfinite(reward).all()
            assert obs['actor'].shape==(8,37) and obs['critic'].shape==(8,45)
            assert torch.equal(obs['actor'][:,-1:],term.command)
            assert torch.equal(obs['critic'][:,-1:],term.command)
            done_count+=int(done.sum())
            if done.any():reset_steps.append({'step':env.common_step_counter,'resets':int(done.sum())})
    up_to_down=sum(s['from']==1 for s in report['switches'])
    down_to_up=sum(s['from']==-1 for s in report['switches'])
    assert up_to_down>0 and down_to_up>0
    report.update({'stand_to_snake':up_to_down,'snake_to_stand':down_to_up,'environment_resets':done_count,'reset_steps':reset_steps,'physical_state_and_targets_preserved':True,'old_command_reward_and_metrics_new_command_observations':True,'completed_since_last_reset':term.metrics['completed_transitions'].tolist()})
    Path('outputs/performance/archived_bidirectional_recheck.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',json.dumps({k:v for k,v in report.items() if k!='switches'}),flush=True)
finally:env.close()
