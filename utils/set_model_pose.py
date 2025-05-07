import gz.transport13 as transport
import gz.msgs10 as msgs

def set_model_pose():
    # Create Ignition Transport node
    node = transport.Node()

    # Connect to the world control service
    control_service = '/world/grass/control'
    node.Subscribe(control_service, msgs.String())

    # Set the desired model name
    model_name = "drone0"

    # Set the desired pose (position and orientation)
    desired_pose = ignition.msgs.Pose()
    desired_pose.position.x = 1.0  # Set desired x position
    desired_pose.position.y = 2.0  # Set desired y position
    desired_pose.position.z = 0.5  # Set desired z position
    desired_pose.orientation.x = 0.0  # Set desired x orientation
    desired_pose.orientation.y = 0.0  # Set desired y orientation
    desired_pose.orientation.z = 0.0  # Set desired z orientation
    desired_pose.orientation.w = 1.0  # Set desired w orientation

    # Create a request message
    request_msg = ignition.msgs.String()
    request_msg.data = f'set model {model_name} pose {desired_pose.SerializeToString()}'

    # Publish the request to the world control service
    node.Request(control_service, request_msg)

if __name__ == '__main__':
    try:
        set_model_pose()
    except transport.TransportError as e:
        print(f"Transport error: {e}"