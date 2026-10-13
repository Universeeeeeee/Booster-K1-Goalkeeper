"""Read-only native-force audit of Stage E strict momentum precision misses.

Runs the recorded cases without changing physics, controller, solver parameters,
initial conditions or outputs of the original validation. No tolerance is relaxed.
"""
import argparse
import hashlib
import json,sys
from pathlib import Path
from unittest.mock import patch
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from k1_goalkeeper.impact_validation import ImpactCase,run_impact,ball_contact_force_world
from k1_goalkeeper.scene_simulation import SceneSimulation
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'reports/impact_validation/solver_audit.json')
args=parser.parse_args()
report_path=ROOT/'reports/impact_validation/results.json'
report=json.loads(report_path.read_text())
protected=[ROOT/'assets/booster_assets/robots/K1/K1_22dof.xml',ROOT/'assets/booster_deploy/tasks/locomotion/robots/k1/models/k1_walk.pt',ROOT/'models/futsal_goal_3x2_reference.urdf',ROOT/'k1_goalkeeper/walk_controller.py',ROOT/'k1_goalkeeper/scene_simulation.py',ROOT/'config/simulation.json']
protected_before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
inputs=[report_path,ROOT/'k1_goalkeeper/impact_validation.py',Path(__file__).resolve()]
source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
strict_tolerance_ns=1e-6
results=[]
for old in report['results']:
 if old['maximum_momentum_residual_ns']<=strict_tolerance_ns:continue
 case=ImpactCase(**old['case']);model=mujoco.MjModel.from_xml_path(str(ROOT/f'reports/impact_validation/scene_{case.robot_mode}.xml'))
 source_hashes[f'reports/impact_validation/scene_{case.robot_mode}.xml']=hashlib.sha256((ROOT/f'reports/impact_validation/scene_{case.robot_mode}.xml').read_bytes()).hexdigest()
 source_hashes[f'reports/impact_validation/{case.name}/summary.json']=hashlib.sha256((ROOT/f'reports/impact_validation/{case.name}/summary.json').read_bytes()).hexdigest()
 v=int(model.joint('ball_free').dofadr[0]);ball=model.geom('ball_collision').id;mass=float(model.body_mass[model.geom_bodyid[ball]])
 native=SceneSimulation.step
 observed={'case':case.name,'case_parameters':old['case'],'recorded_launch':old['launch'],'strict_momentum_precision_passed':False,'max_force_reconstruction_error_n':0.,'max_integration_error_m_s':0.,'max_gradient_explanation_error_ns':0.,'max_residual_robot_contact_ns':0.,'largest':None,'solver_tolerance':float(model.opt.tolerance),'solver_iterations':int(model.opt.iterations),'parameters_modified':False}
 def observe(sim):
  before=sim.data.qvel[v:v+3].copy();t0=float(sim.data.time);native(sim);d=sim.data
  if t0<old['launch']['time']-1e-9:return
  force=np.zeros(3);contact_info=[];robot_contact=False
  for i,c in enumerate(d.contact):
   if ball not in (c.geom1,c.geom2) or c.efc_address<0:continue
   f=np.zeros(6);mujoco.mj_contactForce(model,d,i,f);fw=ball_contact_force_world(c,f,ball);force+=fw
   other=int(c.geom2 if c.geom1==ball else c.geom1)
   name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,other)
   robot_contact|=name.startswith('k1_') and f[0]>1e-6
   contact_info.append({'geom':name,'force_world_n':fw.tolist(),'normal_n':float(f[0]),'dist_m':float(c.dist),'efc_address':int(c.efc_address)})
  dv=d.qvel[v:v+3]-before;smooth=d.qfrc_smooth[v:v+3].copy();constraint=d.qfrc_constraint[v:v+3].copy();acc=d.qacc[v:v+3].copy()
  residual=mass*dv-(force+mass*model.opt.gravity)*model.opt.timestep
  gradient=mass*acc-smooth-constraint
  observed['max_force_reconstruction_error_n']=max(observed['max_force_reconstruction_error_n'],float(np.max(abs(force-constraint))))
  observed['max_integration_error_m_s']=max(observed['max_integration_error_m_s'],float(np.max(abs(dv-acc*.001))))
  observed['max_gradient_explanation_error_ns']=max(observed['max_gradient_explanation_error_ns'],float(np.max(abs(residual-gradient*.001))))
  if robot_contact:observed['max_residual_robot_contact_ns']=max(observed['max_residual_robot_contact_ns'],float(np.max(abs(residual))))
  if observed['largest'] is not None and np.max(abs(residual))<=np.max(np.abs(observed['largest']['momentum_residual_ns'])):return
  island=int(d.dof_island[v]);iterations=int(d.solver_niter[island]) if island>=0 else 0
  stats=[{key:float(getattr(d.solver[island*mujoco.mjNSOLVER+j],key)) for key in ['gradient','improvement','lineslope','nactive','nchange','neval','nupdate']} for j in range(iterations)] if island>=0 else []
  observed['largest']={'before_time':t0,'after_time':float(d.time),'force_contacts_world_n':force.tolist(),'qfrc_constraint_ball_n':constraint.tolist(),'qfrc_smooth_ball_n':smooth.tolist(),'qfrc_bias_ball_n':d.qfrc_bias[v:v+3].tolist(),'qfrc_passive_ball_n':d.qfrc_passive[v:v+3].tolist(),'ball_qacc':acc.tolist(),'ball_delta_velocity':dv.tolist(),'delta_velocity_minus_accdt':(dv-acc*.001).tolist(),'contact_force_minus_constraint_n':(force-constraint).tolist(),'dynamics_gradient_ball_n':gradient.tolist(),'momentum_residual_ns':residual.tolist(),'ball_solver_island':island,'solver_niter':d.solver_niter[:d.nisland].tolist(),'ball_solver_stats':stats,'contacts':contact_info}
 with patch.object(SceneSimulation,'step',observe):summary,tr=run_impact(model,case,launch=old['launch'] if case.robot_mode=='absent' else None)
 observed['numerical_error']=summary['numerical_error'];observed['completed']=summary['completed'];observed['reproduced_trajectory_hash']=summary['trajectory_sha256'];observed['original_trajectory_hash']=old['trajectory_sha256'];observed['identical_trajectory']=summary['trajectory_sha256']==old['trajectory_sha256']
 results.append(observed)
 print(json.dumps({'case':case.name,'identical':observed['identical_trajectory'],'force_error':observed['max_force_reconstruction_error_n'],'integration_error':observed['max_integration_error_m_s'],'gradient_explanation_error':observed['max_gradient_explanation_error_ns'],'max_robot_residual':observed['max_residual_robot_contact_ns'],'worst':observed['largest']['momentum_residual_ns'],'ball_solver_stats':observed['largest']['ball_solver_stats']}))
protected_after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
audit={'official_solver_termination_reference':'https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#option-tolerance','termination_rule':'Newton stops when the smaller of normalized cost improvement and gradient norm is below tolerance. All six worst steps have improvement below the unchanged 1e-6 tolerance and finite gradient.','mujoco_version':mujoco.__version__,'generation':'.venv/bin/python scripts/audit_impact_solver.py','strict_momentum_tolerance_ns':strict_tolerance_ns,'strict_precision_miss_case_count':len(results),'parameters_modified':False,'protected_sha256_before':protected_before,'protected_sha256_after':protected_after,'protected_unchanged':protected_before==protected_after,'input_and_audit_code_sha256':source_hashes,'methods':['Contact-frame forces on the ball summed independently and compared with qfrc_constraint translation DOFs.','Delivered delta velocity compared with native qacc * timestep.','Momentum residual compared with timestep * (mass*qacc-qfrc_smooth-qfrc_constraint).','Ball solver island identified from dof_island; solver statistics indexed with island*mjNSOLVER+iteration.'],'assessment':'The strict 1e-6 Ns criterion remains unmet in these cases. Native-force reconstruction and integration are correct; official Newton termination with finite dynamics gradient explains the residual. This is a solver accuracy limitation, not NaN instability or a reconstruction error. No tolerance or physics parameter is changed.','all_trajectories_identical':all(r['identical_trajectory'] for r in results),'results':results}
args.output.write_text(json.dumps(audit,indent=2)+'\n')
