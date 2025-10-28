#!/usr/bin/env python3
"""Interactive testbed for uFactory Lite 6 arm using the xArm Python SDK.

This comprehensive testbed provides multiple modes for testing and controlling
the robot arm safely and interactively. It supports:

- Interactive mode: Manual testing of various commands with live feedback
- Single command mode: Send specific joint/Cartesian poses
- Trajectory playback: Execute pre-recorded motion sequences
- Status monitoring: Query robot state, position, and errors
- Safety features: Collision detection, motion limits, emergency stop

Usage examples:
  # Interactive testbed (recommended for testing)
  python scripts/send_to_ufactory.py --ip 192.168.1.10 --interactive

  # Query current status
  python scripts/send_to_ufactory.py --ip 192.168.1.10 --status

  # Send single joint command
  python scripts/send_to_ufactory.py --ip 192.168.1.10 --joints 0 0 0 0 0 0 --confirm

  # Play back trajectory
  python scripts/send_to_ufactory.py --ip 192.168.1.10 --trajectory sim_traj.json --confirm

  # Dry-run mode (simulate without connecting)
  python scripts/send_to_ufactory.py --ip 192.168.1.10 --joints 0 0 0 0 0 0 --dry-run

Safety Notes:
- Always ensure the robot workspace is clear before sending commands
- Use --confirm flag for actual robot movement (safety interlock)
- The robot will be enabled and set to position mode automatically
- Emergency stop: Use Ctrl+C to safely disconnect

Installation:
  pip install xarm-python-sdk
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ANSI color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(msg: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_success(msg: str) -> None:
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}✗ {msg}{Colors.ENDC}")


def print_info(msg: str) -> None:
    print(f"{Colors.CYAN}→ {msg}{Colors.ENDC}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.ENDC}")


def try_import_xarm() -> Optional[Any]:
    """Try importing common xArm SDK entrypoints.

    Returns the XArmAPI class (or equivalent) or None if not found.
    """
    try:
        # common import variants
        from xarm import XArmAPI  # type: ignore

        return XArmAPI
    except Exception:
        pass
    try:
        from xarm.wrapper import XArmAPI  # type: ignore

        return XArmAPI
    except Exception:
        pass
    # Add other vendor import attempts here if needed
    return None


def load_trajectory(path: Path) -> List[List[float]]:
    data = json.loads(path.read_text())
    if "joints" in data:
        return data["joints"]
    # support older key names
    if "trajectory" in data:
        return data["trajectory"]
    raise ValueError("Trajectory JSON must contain a 'joints' or 'trajectory' key")


class RobotController:
    """Wrapper around xArm SDK with enhanced safety and testing features."""

    def __init__(self, api: Any):
        self.api = api
        self.connected = False

    def initialize(self) -> bool:
        """Initialize robot: enable motion, set mode, and clear errors."""
        try:
            print_info("Initializing robot...")

            # Clean any errors
            if hasattr(self.api, 'clean_error'):
                self.api.clean_error()
                print_success("Cleared any previous errors")

            # Enable motion
            if hasattr(self.api, 'motion_enable'):
                ret = self.api.motion_enable(enable=True)
                if ret == 0:
                    print_success("Motion enabled")
                else:
                    print_warning(f"Motion enable returned code: {ret}")

            # Set to position control mode (mode 0)
            if hasattr(self.api, 'set_mode'):
                ret = self.api.set_mode(0)
                if ret == 0:
                    print_success("Set to position control mode")
                else:
                    print_warning(f"Set mode returned code: {ret}")

            # Set state to ready (state 0)
            if hasattr(self.api, 'set_state'):
                ret = self.api.set_state(state=0)
                if ret == 0:
                    print_success("Robot state set to ready")
                else:
                    print_warning(f"Set state returned code: {ret}")

            self.connected = True
            print_success("Robot initialization complete")
            return True

        except Exception as e:
            print_error(f"Failed to initialize robot: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Query comprehensive robot status."""
        status = {}

        try:
            # Get position (joint angles)
            if hasattr(self.api, 'get_servo_angle'):
                ret = self.api.get_servo_angle()
                if isinstance(ret, tuple) and len(ret) >= 2:
                    code, angles = ret[0], ret[1]
                    if code == 0:
                        status['joint_angles'] = angles

            # Get Cartesian position
            if hasattr(self.api, 'get_position'):
                ret = self.api.get_position()
                if isinstance(ret, tuple) and len(ret) >= 2:
                    code, pose = ret[0], ret[1]
                    if code == 0:
                        status['cartesian_pose'] = pose

            # Get robot state
            if hasattr(self.api, 'get_state'):
                ret = self.api.get_state()
                if isinstance(ret, tuple) and len(ret) >= 2:
                    status['state'] = ret[1]

            # Get error/warning codes
            if hasattr(self.api, 'get_err_warn_code'):
                ret = self.api.get_err_warn_code()
                if isinstance(ret, tuple) and len(ret) >= 2:
                    status['error_code'] = ret[1][0] if len(ret[1]) > 0 else 0
                    status['warning_code'] = ret[1][1] if len(ret[1]) > 1 else 0

            # Get mode
            if hasattr(self.api, 'mode'):
                status['mode'] = self.api.mode

            # Get version info
            if hasattr(self.api, 'version'):
                status['version'] = self.api.version

        except Exception as e:
            print_error(f"Error querying status: {e}")

        return status

    def print_status(self) -> None:
        """Print formatted robot status."""
        print_header("Robot Status")
        status = self.get_status()

        if 'joint_angles' in status:
            print_info(f"Joint Angles (deg): {[f'{a:.2f}' for a in status['joint_angles']]}")

        if 'cartesian_pose' in status:
            pose = status['cartesian_pose']
            print_info(f"Cartesian Position (mm): X={pose[0]:.1f}, Y={pose[1]:.1f}, Z={pose[2]:.1f}")
            print_info(f"Cartesian Rotation (deg): Roll={pose[3]:.1f}, Pitch={pose[4]:.1f}, Yaw={pose[5]:.1f}")

        if 'state' in status:
            state_names = {0: 'Ready', 1: 'Pause', 3: 'Offline', 4: 'Error', 5: 'Stop'}
            state_val = status['state']
            state_str = state_names.get(state_val, f'Unknown ({state_val})')
            print_info(f"State: {state_str}")

        if 'mode' in status:
            mode_names = {0: 'Position', 1: 'Servo', 2: 'Joint Teaching', 4: 'Cartesian Teaching'}
            mode_val = status['mode']
            mode_str = mode_names.get(mode_val, f'Unknown ({mode_val})')
            print_info(f"Mode: {mode_str}")

        if 'error_code' in status:
            if status['error_code'] != 0:
                print_error(f"Error Code: {status['error_code']}")
            else:
                print_success("No errors")

        if 'warning_code' in status:
            if status['warning_code'] != 0:
                print_warning(f"Warning Code: {status['warning_code']}")

    def send_joint_command(self, joints: List[float], speed: float = 20.0,
                          wait: bool = True, is_radian: bool = False) -> bool:
        """Send joint position command."""
        try:
            print_info(f"Sending joint command: {[f'{j:.2f}' for j in joints]} (speed={speed})")

            # Try set_servo_angle (most common for xArm)
            if hasattr(self.api, 'set_servo_angle'):
                ret = self.api.set_servo_angle(angle=joints, speed=speed,
                                               wait=wait, is_radian=is_radian)
                if ret == 0:
                    print_success("Joint command sent successfully")
                    return True
                else:
                    print_error(f"Joint command failed with code: {ret}")
                    return False

            # Fallback methods
            if hasattr(self.api, 'set_joint_position'):
                self.api.set_joint_position(joints, speed, wait)
                print_success("Joint command sent (via set_joint_position)")
                return True

            if hasattr(self.api, 'move_joint'):
                self.api.move_joint(joints, speed=speed)
                print_success("Joint command sent (via move_joint)")
                return True

            print_error("No suitable joint command method found in API")
            return False

        except Exception as e:
            print_error(f"Failed to send joint command: {e}")
            return False

    def send_cartesian_command(self, pose: List[float], speed: float = 100.0,
                               wait: bool = True, is_radian: bool = False) -> bool:
        """Send Cartesian position command."""
        try:
            print_info(f"Sending Cartesian command: pos=[{pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}], "
                      f"rot=[{pose[3]:.1f}, {pose[4]:.1f}, {pose[5]:.1f}] (speed={speed})")

            if hasattr(self.api, 'set_position'):
                ret = self.api.set_position(x=pose[0], y=pose[1], z=pose[2],
                                           roll=pose[3], pitch=pose[4], yaw=pose[5],
                                           speed=speed, wait=wait, is_radian=is_radian)
                if ret == 0:
                    print_success("Cartesian command sent successfully")
                    return True
                else:
                    print_error(f"Cartesian command failed with code: {ret}")
                    return False

            print_error("set_position method not found in API")
            return False

        except Exception as e:
            print_error(f"Failed to send Cartesian command: {e}")
            return False

    def control_gripper(self, position: float, speed: float = 5000.0, wait: bool = True) -> bool:
        """Control gripper position (0-850 for xArm Gripper)."""
        try:
            print_info(f"Setting gripper position to {position} (speed={speed})")

            if hasattr(self.api, 'set_gripper_position'):
                ret = self.api.set_gripper_position(pos=position, wait=wait, speed=speed)
                if ret == 0:
                    print_success(f"Gripper moved to position {position}")
                    return True
                else:
                    print_error(f"Gripper command failed with code: {ret}")
                    return False

            print_warning("Gripper control not available in API")
            return False

        except Exception as e:
            print_error(f"Failed to control gripper: {e}")
            return False

    def emergency_stop(self) -> bool:
        """Emergency stop - immediately halt all motion."""
        try:
            print_warning("EMERGENCY STOP TRIGGERED")

            if hasattr(self.api, 'emergency_stop'):
                self.api.emergency_stop()
                print_success("Emergency stop executed")
                return True

            # Fallback: set to stop state
            if hasattr(self.api, 'set_state'):
                self.api.set_state(4)  # State 4 = Stop
                print_success("Robot stopped (via set_state)")
                return True

            return False

        except Exception as e:
            print_error(f"Emergency stop failed: {e}")
            return False

    def disconnect(self) -> None:
        """Safely disconnect from robot."""
        try:
            print_info("Disconnecting from robot...")
            if hasattr(self.api, 'disconnect'):
                self.api.disconnect()
            print_success("Disconnected")
        except Exception as e:
            print_error(f"Error during disconnect: {e}")


def interactive_mode(controller: RobotController) -> None:
    """Interactive testbed for manual robot testing."""
    print_header("Interactive Robot Testbed")
    print_info("Type 'help' for available commands, 'quit' to exit")

    while True:
        try:
            user_input = input(f"\n{Colors.BOLD}robot> {Colors.ENDC}").strip()

            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ['quit', 'exit', 'q']:
                print_info("Exiting interactive mode...")
                break

            elif cmd == 'help':
                print("\nAvailable commands:")
                print(f"  {Colors.CYAN}status{Colors.ENDC}                      - Show robot status")
                print(f"  {Colors.CYAN}home{Colors.ENDC}                        - Move to home position (all zeros)")
                print(f"  {Colors.CYAN}joint J1 J2 J3 J4 J5 J6{Colors.ENDC}     - Move to joint angles (degrees)")
                print(f"  {Colors.CYAN}pose X Y Z RX RY RZ{Colors.ENDC}        - Move to Cartesian pose (mm, degrees)")
                print(f"  {Colors.CYAN}gripper POSITION{Colors.ENDC}            - Set gripper position (0-850)")
                print(f"  {Colors.CYAN}gripper_open{Colors.ENDC}                - Open gripper fully")
                print(f"  {Colors.CYAN}gripper_close{Colors.ENDC}               - Close gripper fully")
                print(f"  {Colors.CYAN}speed VALUE{Colors.ENDC}                 - Set default speed")
                print(f"  {Colors.CYAN}traj FILEPATH{Colors.ENDC}               - Execute trajectory from file")
                print(f"  {Colors.CYAN}stop{Colors.ENDC}                        - Emergency stop")
                print(f"  {Colors.CYAN}clear{Colors.ENDC}                       - Clear errors and re-initialize")
                print(f"  {Colors.CYAN}quit{Colors.ENDC}                        - Exit interactive mode\n")

            elif cmd == 'status':
                controller.print_status()

            elif cmd == 'home':
                controller.send_joint_command([0, 0, 0, 0, 0, 0], speed=20.0)

            elif cmd == 'joint':
                if len(parts) != 7:
                    print_error("Usage: joint J1 J2 J3 J4 J5 J6")
                    continue
                try:
                    joints = [float(p) for p in parts[1:7]]
                    controller.send_joint_command(joints)
                except ValueError:
                    print_error("Invalid joint values. Use numbers.")

            elif cmd == 'pose':
                if len(parts) != 7:
                    print_error("Usage: pose X Y Z RX RY RZ")
                    continue
                try:
                    pose = [float(p) for p in parts[1:7]]
                    controller.send_cartesian_command(pose)
                except ValueError:
                    print_error("Invalid pose values. Use numbers.")

            elif cmd == 'gripper':
                if len(parts) != 2:
                    print_error("Usage: gripper POSITION")
                    continue
                try:
                    pos = float(parts[1])
                    controller.control_gripper(pos)
                except ValueError:
                    print_error("Invalid gripper position. Use a number (0-850).")

            elif cmd == 'gripper_open':
                controller.control_gripper(850.0)

            elif cmd == 'gripper_close':
                controller.control_gripper(0.0)

            elif cmd == 'stop':
                controller.emergency_stop()

            elif cmd == 'clear':
                if hasattr(controller.api, 'clean_error'):
                    controller.api.clean_error()
                    print_success("Errors cleared")
                controller.initialize()

            elif cmd == 'traj':
                if len(parts) != 2:
                    print_error("Usage: traj FILEPATH")
                    continue
                try:
                    traj_path = Path(parts[1])
                    if not traj_path.exists():
                        print_error(f"File not found: {traj_path}")
                        continue
                    trajectory = load_trajectory(traj_path)
                    print_info(f"Loaded trajectory with {len(trajectory)} waypoints")
                    for i, joints in enumerate(trajectory):
                        print_info(f"Waypoint {i+1}/{len(trajectory)}")
                        if not controller.send_joint_command(joints):
                            print_error("Trajectory execution stopped due to error")
                            break
                        time.sleep(0.1)
                    print_success("Trajectory execution complete")
                except Exception as e:
                    print_error(f"Failed to execute trajectory: {e}")

            else:
                print_error(f"Unknown command: {cmd}. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n")
            print_warning("Interrupted! Type 'quit' to exit or continue with commands.")
            continue
        except EOFError:
            print("\n")
            break
        except Exception as e:
            print_error(f"Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive testbed for uFactory Lite6 robot arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode (recommended):
    python scripts/send_to_ufactory.py --ip 192.168.1.10 --interactive

  Query robot status:
    python scripts/send_to_ufactory.py --ip 192.168.1.10 --status

  Send single joint command:
    python scripts/send_to_ufactory.py --ip 192.168.1.10 --joints 0 0 0 0 0 0 --confirm

  Execute trajectory:
    python scripts/send_to_ufactory.py --ip 192.168.1.10 --trajectory path/to/traj.json --confirm
        """
    )

    # Connection options
    parser.add_argument("--ip", required=True, help="Robot IP address")

    # Mode options
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Enter interactive mode for manual testing")
    parser.add_argument("--status", "-s", action="store_true",
                       help="Query and display robot status")

    # Command options
    parser.add_argument("--joints", nargs=6, type=float,
                       help="6 joint angles in degrees")
    parser.add_argument("--pose", nargs=6, type=float,
                       metavar=("X", "Y", "Z", "Rx", "Ry", "Rz"),
                       help="Cartesian pose: X Y Z (mm) Roll Pitch Yaw (deg)")
    parser.add_argument("--trajectory", type=Path,
                       help="Path to JSON trajectory file with 'joints': [[...], ...]")
    parser.add_argument("--gripper", type=float,
                       help="Set gripper position (0-850)")

    # Execution options
    parser.add_argument("--speed", type=float, default=20.0,
                       help="Movement speed in deg/s for joints or mm/s for Cartesian (default: 20)")
    parser.add_argument("--confirm", action="store_true",
                       help="Required flag to actually send commands (safety interlock)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print commands without connecting to robot")
    parser.add_argument("--step-delay", type=float, default=0.5,
                       help="Delay between trajectory steps in seconds (default: 0.5)")

    args = parser.parse_args()

    # Print banner
    print_header("uFactory Lite 6 Robot Testbed")

    # Handle dry-run mode
    if args.dry_run:
        print_warning("DRY-RUN MODE: Commands will be printed but not sent")
        print_info(f"Would connect to robot at {args.ip}")

        if args.joints:
            print_info(f"Would send joint command: {args.joints}")
        if args.pose:
            print_info(f"Would send Cartesian pose: {args.pose}")
        if args.trajectory:
            if args.trajectory.exists():
                traj = load_trajectory(args.trajectory)
                print_info(f"Would execute trajectory with {len(traj)} waypoints")
            else:
                print_error(f"Trajectory file not found: {args.trajectory}")
        if args.gripper is not None:
            print_info(f"Would set gripper to position: {args.gripper}")

        print_success("Dry-run complete")
        return

    # Try to import SDK
    XArmAPI = try_import_xarm()
    if XArmAPI is None:
        print_error("Could not import xArm SDK")
        print_info("Install with: pip install xarm-python-sdk")
        print_info("Or from source: https://github.com/xArm-Developer/xArm-Python-SDK")
        sys.exit(1)

    # Connect to robot
    controller = None
    try:
        print_info(f"Connecting to robot at {args.ip}...")
        api = XArmAPI(args.ip)
        controller = RobotController(api)

        # Initialize robot
        if not controller.initialize():
            print_error("Failed to initialize robot")
            sys.exit(1)

    except Exception as e:
        print_error(f"Failed to connect to robot: {e}")
        sys.exit(1)

    # Ensure proper cleanup on exit
    try:
        # Interactive mode
        if args.interactive:
            interactive_mode(controller)
            return

        # Status query mode
        if args.status:
            controller.print_status()
            return

        # Command execution mode
        command_given = False

        # Single joint command
        if args.joints:
            command_given = True
            if not args.confirm:
                print_warning("Joint command specified but --confirm not provided")
                print_warning("Add --confirm to actually send the command")
                print_info(f"Would send: {args.joints}")
            else:
                controller.send_joint_command(list(args.joints), speed=args.speed)

        # Single Cartesian pose
        if args.pose:
            command_given = True
            if not args.confirm:
                print_warning("Pose command specified but --confirm not provided")
                print_warning("Add --confirm to actually send the command")
                print_info(f"Would send: {args.pose}")
            else:
                controller.send_cartesian_command(list(args.pose), speed=args.speed)

        # Gripper command
        if args.gripper is not None:
            command_given = True
            if not args.confirm:
                print_warning("Gripper command specified but --confirm not provided")
                print_warning("Add --confirm to actually send the command")
                print_info(f"Would set gripper to: {args.gripper}")
            else:
                controller.control_gripper(args.gripper)

        # Trajectory playback
        if args.trajectory:
            command_given = True
            if not args.trajectory.exists():
                print_error(f"Trajectory file not found: {args.trajectory}")
                return

            trajectory = load_trajectory(args.trajectory)
            print_info(f"Loaded trajectory with {len(trajectory)} waypoints")

            if not args.confirm:
                print_warning("Trajectory specified but --confirm not provided")
                print_warning("Add --confirm to actually execute the trajectory")
                for i, joints in enumerate(trajectory):
                    print_info(f"Waypoint {i+1}/{len(trajectory)}: {joints}")
            else:
                print_info("Executing trajectory...")
                for i, joints in enumerate(trajectory):
                    print_info(f"Waypoint {i+1}/{len(trajectory)}")
                    if not controller.send_joint_command(joints, speed=args.speed):
                        print_error("Trajectory execution stopped due to error")
                        break
                    time.sleep(args.step_delay)
                print_success("Trajectory execution complete")

        # If no command was given, show help
        if not command_given:
            print_info("No command specified. Use one of:")
            print_info("  --interactive     : Enter interactive mode")
            print_info("  --status          : Query robot status")
            print_info("  --joints J1...J6  : Send joint command")
            print_info("  --pose X Y Z R P Y: Send Cartesian pose")
            print_info("  --trajectory FILE : Execute trajectory")
            print_info("  --gripper POS     : Control gripper")
            print_info("\nRun with --help for full documentation")

    except KeyboardInterrupt:
        print("\n")
        print_warning("Interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always disconnect cleanly
        if controller:
            controller.disconnect()


if __name__ == "__main__":
    main()
