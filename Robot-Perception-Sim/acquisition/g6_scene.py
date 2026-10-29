"""Full G1 K1 scene with an official head-mounted ZED and frozen G4 replay.

No policy, physics response, or camera hardware-extrinsic claim. USD uses row
matrices; public optical transforms here use column vectors (+X right,+Y down,
+Z forward). The existing mount config is an explicit virtual assumption.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HEAD_PATH = "/World/G1/Bodies/b3_aahead_pitch_link/b5_head_booster_stereo_rgb_link"
OPTICAL_FROM_USD = np.diag([1., -1., -1., 1.])
HEAD_POSES = {"neutral": (0., 0.), "yaw_pitch": (.2, .1)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest_sample(samples, requested):
    times = np.asarray([s["time"] for s in samples])
    if not np.isfinite(requested):
        raise ValueError("Requested time must be finite")
    index = int(np.argmin(np.abs(times - requested)))
    return samples[index], index


def sphere_fov(center, radius, fx=370.8, fy=370.8, width=960, height=600):
    """Conservative sphere/frustum-plane test; independently of detector output."""
    center = np.asarray(center, dtype=float)
    x, y, z = center
    planes = [np.array([0.,0.,1.]), np.array([fx,0,width/2]), np.array([-fx,0,width/2]),
              np.array([0,fy,height/2]), np.array([0,-fy,height/2])]
    distances = [float(n @ center / np.linalg.norm(n)) for n in planes]
    return {"in_fov": all(d >= -radius for d in distances),
            "fully_in_fov": all(d >= radius for d in distances),
            "center_optical_m": center.tolist(), "frustum_plane_distances_m": distances,
            "basis": "sphere against optical pinhole frustum planes; independent of image detector"}


class G6Scene:
    def __init__(self, stage, g1_usd, camera_usd, trajectory_paths, mount_config_path,
                 head_pose="neutral"):
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        self.Gf, self.Usd, self.UsdGeom = Gf, Usd, UsdGeom
        self.stage = stage
        self.g1_usd, self.camera_usd = Path(g1_usd), Path(camera_usd)
        self.mount_config_path = Path(mount_config_path)
        self.mount_config = json.loads(self.mount_config_path.read_text())
        root = stage.DefinePrim("/World/G1", "Xform")
        root.GetReferences().AddReference(str(self.g1_usd.resolve()))
        self.yaw_path = "/World/G1/Bodies/b2_aahead_yaw_link"
        self.pitch_path = "/World/G1/Bodies/b3_aahead_pitch_link"
        self.ball_path = "/World/G1/Bodies/b28_ball"
        for path in (HEAD_PATH, self.yaw_path, self.pitch_path, self.ball_path):
            if not stage.GetPrimAtPath(path):
                raise ValueError(f"Required frozen G1 prim missing: {path}")
        self.trunk_world = self._world("/World/G1/Bodies/b1_trunk")
        self.joint_frames = []
        for name in ("aahead_yaw_joint", "aahead_pitch_joint"):
            joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/G1/Joints/" + name))
            self.joint_frames.append((self._pose(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()),
                                      self._pose(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get()),
                                      np.deg2rad(joint.GetLowerLimitAttr().Get()),
                                      np.deg2rad(joint.GetUpperLimitAttr().Get())))
        # Frozen geometry is a visual replay fixture. Disable collision response
        # and source joints; source poses/geometries/colors remain unchanged.
        self.frozen_body_paths = []
        self.zero_opacity_visibility_overrides = []
        for prim in list(Usd.PrimRange(root)):
            if prim.IsA(UsdGeom.Gprim):
                opacity_attr = UsdGeom.Gprim(prim).GetDisplayOpacityAttr()
                opacity = opacity_attr.Get()
                if opacity_attr.HasAuthoredValueOpinion() and opacity and all(value == 0 for value in opacity):
                    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set("invisible")
                    self.zero_opacity_visibility_overrides.append({"path": str(prim.GetPath()),
                        "source_display_opacity": list(opacity), "g6_visibility": "invisible"})
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                self.frozen_body_paths.append(str(prim.GetPath()))
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint) or prim.IsA(UsdPhysics.Scene):
                prim.SetActive(False)
        self.camera_root_path = HEAD_PATH + "/ZEDMount/ZED"
        self.left_camera_path = self.camera_root_path + "/base_link/ZED_X/CameraLeft"
        self.right_camera_path = self.camera_root_path + "/base_link/ZED_X/CameraRight"
        rig = stage.DefinePrim(self.camera_root_path, "Xform")
        rig.GetReferences().AddReference(str(self.camera_usd.resolve()))
        for prim in Usd.PrimRange(rig):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        head_world = self._world(HEAD_PATH)
        native_left_in_head = np.linalg.inv(head_world) @ self._world(self.left_camera_path) @ OPTICAL_FROM_USD
        # Config offset denotes the stereo midpoint, expressed in the original
        # optical head frame. Left center is half a baseline to optical-left.
        desired_left = np.eye(4)
        desired_left[:3,3] = self.mount_config["mount"]["position_parent_m"]
        desired_left[0,3] -= self.mount_config["baseline_m"] / 2
        if self.mount_config["mount"]["quaternion_parent_from_mount_wxyz"] != [1.,0.,0.,0.]:
            raise ValueError("G6-A audited mount requires the existing identity mount orientation")
        self.head_from_rig = desired_left @ np.linalg.inv(native_left_in_head)
        self._set_matrix(rig, self.head_from_rig)
        for path in (self.left_camera_path, self.right_camera_path):
            cam = UsdGeom.Camera(stage.GetPrimAtPath(path))
            cam.CreateFocalLengthAttr(1.1124)
            cam.CreateHorizontalApertureAttr(2.88)
            cam.CreateVerticalApertureAttr(1.8)
            cam.CreateHorizontalApertureOffsetAttr(0.)
            cam.CreateVerticalApertureOffsetAttr(0.)
        self.trajectory_paths = ({"replay": Path(trajectory_paths)} if isinstance(trajectory_paths, (str,Path))
                                 else {key: Path(value) for key,value in trajectory_paths.items()})
        self.select_trajectory(next(iter(self.trajectory_paths)))
        self.select_head_pose(head_pose)

    def _pose(self, position, quaternion):
        matrix = self.Gf.Matrix4d(1)
        matrix.SetRotate(self.Gf.Rotation(self.Gf.Quatd(float(quaternion.GetReal()),
                         self.Gf.Vec3d(*map(float,quaternion.GetImaginary())))))
        matrix.SetTranslateOnly(self.Gf.Vec3d(*map(float,position)))
        return np.asarray(matrix).T.copy()

    def _world(self, path):
        return np.asarray(self.UsdGeom.XformCache().GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))).T.copy()

    def _set_matrix(self, prim, column_matrix):
        xform = self.UsdGeom.Xformable(prim)
        attr = prim.GetAttribute("xformOp:transform")
        op = self.UsdGeom.XformOp(attr) if attr else xform.AddTransformOp()
        xform.SetXformOpOrder([op])
        op.Set(self.Gf.Matrix4d(np.asarray(column_matrix).T.tolist()))

    @property
    def camera_pose(self):
        """Gf row USD-left-camera pose for the existing G3 marker helpers."""
        return self.Gf.Matrix4d(self._world(self.left_camera_path).T.tolist())

    def select_trajectory(self, name):
        self.trajectory_name = name
        self.trajectory_path = self.trajectory_paths[name]
        self.record = json.loads(self.trajectory_path.read_text())
        self.samples = self.record["trajectory"]
        times = np.array([s["time"] for s in self.samples])
        if not len(times) or not np.all(np.diff(times)>0):
            raise ValueError("Frozen trajectory requires increasing sample times")
        self.duration_s = float(times[-1])

    def select_head_pose(self, name):
        angles = HEAD_POSES[name]
        parent = self.trunk_world.copy()
        for path, angle, (frame0,frame1,lo,hi) in zip((self.yaw_path,self.pitch_path),angles,self.joint_frames):
            if not lo <= angle <= hi:
                raise ValueError("Head pose outside frozen source joint limits")
            c,s = np.cos(angle),np.sin(angle)
            rotation = np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]],dtype=float)
            parent = parent @ frame0 @ rotation @ np.linalg.inv(frame1)
            self._set_matrix(self.stage.GetPrimAtPath(path), parent)
        self.head_pose = name
        self._ray_meshes = None

    def camera_state(self):
        left = self._world(self.left_camera_path) @ OPTICAL_FROM_USD
        right = self._world(self.right_camera_path) @ OPTICAL_FROM_USD
        intrinsics = {}
        for side,path in (("left",self.left_camera_path),("right",self.right_camera_path)):
            cam = self.UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            focal = cam.GetFocalLengthAttr().Get()
            intrinsics[side] = [[960*focal/cam.GetHorizontalApertureAttr().Get(),0,480],
                                [0,600*focal/cam.GetVerticalApertureAttr().Get(),300],[0,0,1]]
        return {"T_world_camera": left.tolist(), "T_world_left_optical": left.tolist(),
                "T_world_right_optical": right.tolist(), "T_left_from_right": (np.linalg.inv(left)@right).tolist(),
                "left_optical_axis_world": left[:3,2].tolist(),
                "right_optical_axis_world": right[:3,2].tolist(),
                "baseline_m": float(np.linalg.norm(right[:3,3]-left[:3,3])),
                "head_pose": self.head_pose, "head_joint_yaw_pitch_rad": list(HEAD_POSES[self.head_pose]),
                "K": intrinsics["left"], "K_per_eye": intrinsics,
                "world_from_usd_camera_row_matrix": {
                    "left": self._world(self.left_camera_path).T.tolist(),
                    "right": self._world(self.right_camera_path).T.tolist()}}

    def author_state(self, t_requested):
        sample,index = nearest_sample(self.samples,t_requested)
        quaternion = sample.get("quaternion_wxyz",[1.,0.,0.,0.])
        pose = self._pose(sample["position"],self.Gf.Quatd(float(quaternion[0]),self.Gf.Vec3d(*quaternion[1:])))
        self._set_matrix(self.stage.GetPrimAtPath(self.ball_path),pose)
        camera = self.camera_state()
        optical = np.linalg.inv(np.asarray(camera["T_world_camera"])) @ np.r_[sample["position"],1]
        state = {"source": "frozen_g4_kinematic_replay", "new_collision_response": False,
                 "trajectory_name": self.trajectory_name, "trajectory_path": str(self.trajectory_path),
                 "trajectory_sha256": sha256(self.trajectory_path), "source_sample_index": index,
                 "requested_source_time_s": float(t_requested), "realized_source_time_s": sample["time"],
                 "source_time_s": sample["time"], "quantization_error_s": sample["time"]-t_requested,
                 "trajectory_requested_time_s": float(t_requested), "trajectory_realized_time_s": sample["time"],
                 "source_trajectory_sha256": sha256(self.trajectory_path),
                 "replay_phase": "preroll_hold" if t_requested<0 else "postroll_hold" if t_requested>self.duration_s else "trajectory",
                 "ball_world_position_m": sample["position"], "ball_center_camera_m": optical[:3].tolist(),
                 "ball_velocity_world_m_s": sample["velocity"], "ball_quaternion_wxyz": quaternion,
                 "ball_radius_m": .11, "ball_appearance": "frozen_g1_plain_white_no_checker_texture",
                 **camera}
        visibility = self.visibility_state(sample["position"])
        state.update(visibility)
        state["visibility_basis"] = visibility["visibility_evidence"]["basis"]
        return state

    def _prepare_mesh_rays(self):
        """Independent triangle-ray evidence from actual visible USD meshes."""
        result = []
        for prim in self.stage.Traverse():
            if not prim.IsA(self.UsdGeom.Mesh) or str(prim.GetPath()).startswith(self.ball_path):
                continue
            imageable = self.UsdGeom.Imageable(prim)
            if imageable.ComputeVisibility() == "invisible":
                continue
            geom = self.UsdGeom.Gprim(prim)
            opacity = geom.GetDisplayOpacityAttr().Get()
            if opacity and max(opacity) <= 0:
                continue
            mesh = self.UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            if not len(points):
                continue
            world = (self._world(str(prim.GetPath())) @ np.c_[points,np.ones(len(points))].T).T[:,:3]
            indices = list(mesh.GetFaceVertexIndicesAttr().Get())
            triangles,offset = [],0
            for n in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset+n]
                triangles.extend((face[0],face[i],face[i+1]) for i in range(1,n-1))
                offset += n
            if not triangles:
                continue
            v = world[np.asarray(triangles)]
            result.append((str(prim.GetPath()), world.min(axis=0),world.max(axis=0),v[:,0],v[:,1]-v[:,0],v[:,2]-v[:,0]))
        self._ray_meshes = result

    def visibility_state(self, ball_world_position):
        left = np.asarray(self.camera_state()["T_world_camera"])
        center = (np.linalg.inv(left) @ np.r_[ball_world_position,1])[:3]
        fov = sphere_fov(center,.11)
        if not fov["in_fov"]:
            return {"expected_visible": False, "visibility_reason": "out_of_fov", "visibility_evidence": fov}
        if self._ray_meshes is None:
            self._prepare_mesh_rays()
        origin = left[:3,3]
        near_clip = float(self.UsdGeom.Camera(self.stage.GetPrimAtPath(self.left_camera_path)).GetClippingRangeAttr().Get()[0])
        offsets = [(0.,0.)]+[(.8*.11*np.cos(a),.8*.11*np.sin(a)) for a in np.linspace(0,2*np.pi,8,endpoint=False)]
        blocked = []
        for x,y in offsets:
            target_c = center + [x,y,-np.sqrt(.11**2-x*x-y*y)]
            if target_c[2] <= near_clip:
                blocked.append(["target_behind_near_clip_plane"])
                continue
            near_fraction = near_clip / target_c[2]
            target = (left @ np.r_[target_c,1])[:3]
            direction = target-origin
            hit_paths=[]
            for path,lo,hi,v0,e1,e2 in self._ray_meshes:
                if np.any(np.maximum(origin,target)<lo) or np.any(np.minimum(origin,target)>hi):
                    continue
                p = np.cross(direction,e2)
                determinant = np.einsum('ij,ij->i',e1,p)
                valid = abs(determinant)>1e-10
                inv = np.divide(1.,determinant,out=np.zeros_like(determinant),where=valid)
                tvec=origin-v0
                u=np.einsum('ij,ij->i',tvec,p)*inv
                q=np.cross(tvec,e1)
                v=q@direction*inv
                distance=np.einsum('ij,ij->i',e2,q)*inv
                if np.any(valid & (u>=0) & (v>=0) & (u+v<=1) & (distance>near_fraction) & (distance<1-1e-6)):
                    hit_paths.append(path)
            blocked.append(hit_paths)
        count=sum(bool(p) for p in blocked)
        near_count = sum(paths == ["target_behind_near_clip_plane"] for paths in blocked)
        mesh_count = count-near_count
        reason = ("near_clip_or_camera_sphere_intersection" if near_count==len(offsets) else
                  "occluded" if mesh_count==len(offsets) else "partially_occluded" if mesh_count else
                  "partially_near_clipped" if near_count else "visible")
        return {"expected_visible": count<len(offsets), "visibility_reason": reason,
                "visibility_evidence": {**fov,"mesh_ray_count": len(offsets),"blocked_mesh_ray_count": count,
                    "blocked_mesh_paths_per_ray": blocked, "near_clip_m": near_clip,
                    "near_clipped_ray_count": near_count, "physical_mesh_blocked_ray_count": mesh_count,
                    "occlusion_basis": "9 sphere surface rays against visible USD Mesh triangles including rig; honor actual USD near clip, finite sampling, non-Mesh primitives omitted"}}

    def metadata(self):
        return {**self.provenance(), "selected_trajectory": self.trajectory_name,
                "selected_trajectory_sha256": sha256(self.trajectory_path),
                "duration_s": self.duration_s, "camera": self.camera_state(),
                "trajectory_sources": {key:{"path":str(path),"sha256":sha256(path)}
                                       for key,path in self.trajectory_paths.items()}}

    def provenance(self):
        return {"g1_scene_sha256": sha256(self.g1_usd), "official_camera_usd_sha256": sha256(self.camera_usd),
                "mount_config_sha256": sha256(self.mount_config_path), "mount_category": "existing_virtual_simulation_assumption",
                "head_optical_prim": HEAD_PATH,"camera_root_path": self.camera_root_path,
                "T_head_optical_from_zed_root": self.head_from_rig.tolist(),
                "robot_mode": "kinematic_frozen_no_walk", "source_collision_enabled": False,
                "zero_opacity_visibility_overrides": self.zero_opacity_visibility_overrides,
                "zero_opacity_override_reason": "honor authored source alpha0 in G6 visual layer; renderer A/B proved invisible ground otherwise occludes turf",
                "ball_appearance": "frozen_g1_plain_white_no_checker_texture", "head_poses": HEAD_POSES}
