# Visual Servoing Guide: Real-Time Camera Feedback

## Question: Can we update our plan in real-time with camera feed?

**Short answer**: Yes! But it depends on your approach.

With a camera fixed to the end-effector (eye-in-hand configuration), you have several options ranging from simple static planning to sophisticated real-time control.

---

## Approach Comparison

| Approach | Real-time Feedback | Accuracy | Computational Cost | Moving Targets |
|----------|-------------------|----------|-------------------|----------------|
| **Static Planning** | ❌ No | Low | Low | ❌ No |
| **Reactive Replanning** | ⚠️ Periodic | Medium | Medium | ⚠️ Slow-moving |
| **Visual Servoing** | ✅ Continuous | High | Low | ✅ Yes |
| **Hybrid** | ✅ Adaptive | High | Medium | ✅ Yes |

---

## 1. Static Planning (What we have now)

```python
# Capture image once at start
depth_frame, color_frame = camera.get_frames()
object_pos = detect_and_localize(depth_frame, color_frame)

# Plan entire path
path = sim.plan_and_execute_cartesian_path(goal_pos=object_pos)

# Execute blindly (camera moves with robot, loses sight of target)
```

**Pros:**
- Simple to implement
- No real-time computation needed
- Good for static targets

**Cons:**
- Cannot adapt to moving targets
- Camera moves during execution, losing visual feedback
- Sensitive to calibration errors

**When to use:** Static objects, well-calibrated system, no disturbances

---

## 2. Reactive Replanning

```python
while not reached_target:
    # Periodically re-detect target
    if time_since_last_plan > replan_interval:
        object_pos = detect_and_localize(camera.get_frames())

        # Replan if target moved significantly
        if position_changed > threshold:
            path = plan_new_path(object_pos)
            execute_partial_path(path)
```

**Pros:**
- Adapts to slow-moving targets
- Uses proven path planning algorithms
- Can avoid obstacles

**Cons:**
- Discrete updates (not truly continuous)
- Planning latency can be problematic
- May be "chasing" a moving target

**When to use:** Slow-moving targets, complex workspace with obstacles

---

## 3. Visual Servoing (True Real-Time Control)

### Position-Based Visual Servoing (PBVS)

```python
while not reached_target:
    # Get target position from camera (every control cycle ~50Hz)
    target_pos_camera = detect_and_localize(camera.get_frames())

    # Compute velocity command based on visual error
    ee_pose = robot.get_ee_pose()
    velocity_cmd = controller.compute_velocity(ee_pose, target_pos_camera)

    # Apply velocity directly (no path planning)
    robot.set_velocity(velocity_cmd)
```

**Pros:**
- True real-time feedback (10-50 Hz)
- Handles moving targets naturally
- Very precise at close range
- Computationally efficient

**Cons:**
- No obstacle avoidance (direct motion)
- Can be unstable if poorly tuned
- May take inefficient paths

**When to use:** Precise manipulation, moving targets, final approach phase

### Image-Based Visual Servoing (IBVS)

```python
while not reached_target:
    # Control directly in image space (pixel errors)
    image_features = detect_features(camera.get_frame())
    desired_features = target_features

    # Compute velocity to minimize feature error
    velocity_cmd = controller.servo_in_image_space(
        image_features, desired_features
    )
```

**Pros:**
- More robust to calibration errors
- Natural for 2D visual tracking
- No 3D reconstruction needed

**Cons:**
- More complex control law
- Potential local minima
- Requires feature tracking

---

## 4. Hybrid Approach (Recommended!)

Combine path planning for coarse motion with visual servoing for precision:

```python
# Phase 1: Coarse approach with path planning
target_pos = detect_target_from_camera()
approach_pos = target_pos - offset  # Stay 10cm away

path = plan_path(current_pos, approach_pos)
execute_path(path)

# Phase 2: Switch to visual servoing for precision
while distance_to_target > tolerance:
    target_pos = detect_target_continuously()
    velocity = compute_servo_command(target_pos)
    apply_velocity(velocity)
```

**Pros:**
- Best of both worlds
- Efficient large motions (planning)
- Precise final approach (servoing)
- Naturally handles moving targets
- Obstacle avoidance when needed

**Cons:**
- More complex implementation
- Need to tune switching threshold

**When to use:** Most real-world applications!

---

## Implementation Examples

### Basic Visual Servoing

```python
from applied_planning.control.visual_servoing import VisualServoController

controller = VisualServoController(
    sim_adapter=sim,
    camera_interface=camera,
    hand_eye_calibration=calibration,
    detection_function=detect_object,
    control_rate=20.0,  # 20 Hz
    replan_threshold=0.02  # 2cm
)

success = controller.servo_to_target(timeout=30.0)
```

### Hybrid Control

```python
from applied_planning.control.visual_servoing import HybridController

controller = HybridController(
    sim_adapter=sim,
    camera_interface=camera,
    hand_eye_calibration=calibration,
    detection_function=detect_object,
    switch_distance=0.1  # Switch to servoing within 10cm
)

success = controller.approach_target()
```

---

## Camera-in-Hand Challenges

With end-effector mounted camera, you face unique challenges:

### 1. **Losing Sight During Motion**
- **Problem**: Camera moves with robot, object can leave field of view
- **Solutions**:
  - Use wide-angle camera
  - Plan paths that keep target in view
  - Use external cameras for global tracking
  - Predict target motion

### 2. **Motion Blur**
- **Problem**: Camera motion causes blurry images
- **Solutions**:
  - Use fast shutter speed
  - Capture images during brief stops
  - Use motion compensation
  - Faster cameras (>60 fps)

### 3. **Changing Perspective**
- **Problem**: Object appearance changes as camera moves
- **Solutions**:
  - Use robust feature detection (SIFT, ORB)
  - Learn multi-view object models
  - Use depth information from RealSense
  - Predict appearance changes

### 4. **Hand-Eye Calibration Errors**
- **Problem**: Errors in camera-to-EE transform accumulate
- **Solutions**:
  - Careful calibration procedure
  - Visual servoing is more robust (uses relative motion)
  - Periodic recalibration
  - Online calibration refinement

---

## Performance Metrics

### Static Planning
- **Accuracy**: ±5-10mm (depends on calibration)
- **Update rate**: Once (0 Hz)
- **Computation**: Low (one-time)
- **Latency**: Planning time (~0.1-2s)

### Reactive Replanning
- **Accuracy**: ±2-5mm
- **Update rate**: 1-5 Hz
- **Computation**: Medium (periodic planning)
- **Latency**: Planning time per update

### Visual Servoing
- **Accuracy**: ±0.5-2mm
- **Update rate**: 10-50 Hz
- **Computation**: Low (proportional control)
- **Latency**: <20ms

### Hybrid
- **Accuracy**: ±0.5-2mm
- **Update rate**: Adaptive (planning → servoing)
- **Computation**: Medium (planning once, then servoing)
- **Latency**: Initial planning, then real-time

---

## Choosing Your Approach

### Use **Static Planning** if:
- Target is stationary
- Environment is well-known and static
- Calibration is very accurate
- Speed > precision

### Use **Reactive Replanning** if:
- Target moves slowly (< 5 cm/s)
- Obstacles may appear/move
- Complex workspace geometry
- Planning capability already exists

### Use **Visual Servoing** if:
- Target is moving
- High precision required (< 2mm)
- Real-time feedback available
- Short distances to target

### Use **Hybrid** if:
- Target may be far away AND moving
- Need both efficiency and precision
- Obstacles in workspace
- Professional/production application

---

## Real-World Example: Pick and Place

```python
# 1. Scan for objects from scanning position
scan_pos = np.array([0.3, 0.0, 0.5])  # High position
sim.plan_and_execute_cartesian_path(scan_pos)

# 2. Detect object from camera
object_pos = detect_object_with_realsense()

# 3. Plan approach (avoid obstacles, fast motion)
approach_pos = object_pos + np.array([0, 0, 0.1])  # 10cm above
sim.plan_and_execute_cartesian_path(approach_pos, planner_type="rrt")

# 4. Switch to visual servoing for precise approach
servo_controller = VisualServoController(...)
servo_controller.servo_to_target(timeout=10.0)

# 5. Grasp
close_gripper()

# 6. Retreat with visual feedback (ensure no collisions)
servo_controller.servo_to_target(goal=approach_pos, timeout=5.0)

# 7. Plan to place location
sim.plan_and_execute_cartesian_path(place_pos, planner_type="rrt")
```

---

## Running the Demos

```bash
# Compare all approaches
python examples/realtime_visual_servoing.py --mode all

# Just visual servoing
python examples/realtime_visual_servoing.py --mode servoing

# Hybrid (recommended)
python examples/realtime_visual_servoing.py --mode hybrid
```

---

## Key Takeaways

1. **Eye-in-hand cameras CAN provide real-time feedback** - you're not limited to planning once
2. **Visual servoing gives true continuous control** at 10-50 Hz
3. **Hybrid approach is best for most applications** - use planning for coarse motion, servoing for precision
4. **Moving targets require continuous feedback** - static planning won't work
5. **Trade-offs exist** between accuracy, speed, computational cost, and robustness

For your RealSense camera application, I recommend starting with the **hybrid approach** - it gives you the best of both worlds!
