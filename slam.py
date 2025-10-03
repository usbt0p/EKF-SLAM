import numpy as np

from robot import Robot, normalize_angle
from plotmap import plotMap, plotEstimate, plotMeasurement, plotError
from ekf import predict, update

## https://www.cs.utexas.edu/~pstone/Courses/393Rfall11/resources/RC09-Quinlan.pdf


# note, ld stands for landmark
def generate_static_landmarks(num_static_lds, map_size):
    ''' Generate random static landmarks within the map size '''
    static_ld_positions = map_size * (np.random.rand(num_static_lds, 2) - 0.5)
    static_ld_ids = np.transpose(
        [np.linspace(0, num_static_lds - 1, num_static_lds, dtype="uint16")]
    )
    static_lds = np.append(static_ld_positions, static_ld_ids, axis=1)
    return static_lds


def generate_circuit_landmarks(num_static_lds, center, radius):
    ''' Generate landmarks in a circular circuit around a center point '''
    angles = np.linspace(0, 2 * np.pi, num_static_lds, endpoint=False)
    static_ld_positions = np.array(
        [center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]
    ).T
    static_ld_ids = np.transpose(
        [np.linspace(0, num_static_lds - 1, num_static_lds, dtype="uint16")]
    )
    static_lds = np.append(static_ld_positions, static_ld_ids, axis=1)
    return static_lds

def generate_acceleration_landmarks(num_static_lds, map_size, rotation_angle=0):
    '''A straightahead road with landmarks on both sides, sor of like a lane'''
    lane_width = 10
    spacing = map_size / (num_static_lds // 2)
    angle_rad = np.deg2rad(rotation_angle) if 'rotation_angle' in locals() else 0

    left_side = np.array(
        [[-map_size / 2 + i * spacing, lane_width / 2] for i in range(num_static_lds // 2)]
    )
    right_side = np.array(
        [[-map_size / 2 + i * spacing, -lane_width / 2] for i in range(num_static_lds // 2)]
    )
    static_ld_positions = np.vstack((left_side, right_side))

    # Rotation matrix
    rot = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    static_ld_positions = static_ld_positions.dot(rot.T)

    static_ld_ids = np.transpose(
        [np.linspace(0, num_static_lds - 1, num_static_lds, dtype="uint16")]
    )
    static_lds = np.append(static_ld_positions, static_ld_ids, axis=1)
    return static_lds


# In[Generate dynamic lds]
def generate_dynamic_landmarks(
    num_dynamic_lds, map_size, num_static_lds, velocity_multiplier=5
):
    dynamic_ld_positions = map_size * (np.random.rand(num_dynamic_lds, 2) - 0.5)
    dynamic_ld_velocities = np.random.rand(num_dynamic_lds, 2) - 0.5
    dynamic_ld_ids = np.transpose(
        [
            np.linspace(
                num_static_lds,
                num_static_lds + num_dynamic_lds - 1,
                num_dynamic_lds,
                dtype="uint16",
            )
        ]
    )
    dynamic_lds = np.append(dynamic_ld_positions, dynamic_ld_ids, axis=1)
    dynamic_lds = np.append(dynamic_lds, dynamic_ld_velocities, axis=1)
    return dynamic_lds


def generate_dynamic_ld_traj(dynamic_lds, num_steps, velocity_multiplier):
    ''' Generate trajectory for dynamic landmarks over num_steps time steps '''
    dynamic_ld_traj = dynamic_lds.copy()
    traj = [dynamic_ld_traj.copy()]
    transition_matrix = np.array(
        [
            [1, 0, 0, velocity_multiplier, 0],
            [0, 1, 0, 0, velocity_multiplier],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ]
    )
    for _ in range(num_steps):
        for i in range(len(dynamic_lds)):
            dynamic_lds[i, :] = transition_matrix.dot(dynamic_lds[i, :].T).T
        traj.append(dynamic_lds.copy())
    return np.dstack(traj)


def curved_robot_inputs(num_steps, step_size, curviness):
    ''' Generate robot inputs with curviness at specific segments, sort of 8 shape'''
    robot_inputs = np.zeros((num_steps, 3))
    robot_inputs[:, 0] = step_size
    robot_inputs[4:12, 1] = curviness
    robot_inputs[18:26, 1] = curviness
    return robot_inputs

def random_robot_inputs(num_steps, step_size, curviness):
    ''' Generate robot inputs with random curviness at each step '''
    return np.append(
        step_size * np.ones((num_steps, 1), dtype="uint8"),
        curviness * np.random.randn(num_steps, 2),
        axis=1,
    )

def circular_robot_inputs(num_steps, step_size, radius):
    ''' Generate robot inputs to follow a circular path with given radius '''
    robot_inputs = np.zeros((num_steps, 3))
    for i in range(num_steps):
        robot_inputs[i, 0] = step_size
        # Set curvature so that the robot turns enough to follow a circle of given radius
        # curvature = step_size / radius, but direction can alternate or be adjusted
        robot_inputs[i, 1] = step_size / radius
    return robot_inputs
        
def straight_robot_inputs(num_steps, step_size):
    ''' Generate robot inputs to follow a straight path '''
    return np.append(
        step_size * np.ones((num_steps, 1), dtype="uint8"),
        np.zeros((num_steps, 2)),
        axis=1,
    )


if __name__ == "__main__":

    # ATTENTION this main part is in testing, some stuff is chaotic and not used

    # generate our environment
    num_static_lds = 70
    map_size = 100
    
    #inner_lds = generate_circuit_landmarks(20, (0, 0), 30)
    outer_lds = generate_circuit_landmarks(50, (0, 0), 50)
    static_lds = outer_lds #np.append(inner_lds, outer_lds, axis=0)

    # for now we dont take dynamic landmarks into account
    # num_dynamic_lds = 0
    # velocity_multiplier = 5
    # dynamic_lds = generate_dynamic_landmarks(num_dynamic_lds, map_size, num_static_lds, velocity_multiplier)

    # Define and initialize robot parameters
    field_of_view = 110 # zed 2i's horizontal FOV
    robot_motion_noise = 5 * np.array([[0.1, 0, 0], 
                                       [0, 0.01, 0], 
                                       [0, 0, 0.01]])
    robot_measurement_noise = np.array([[0.01, 0], 
                                        [0, 0.01]])

    # a triple of (x_position, y_position, orientation(radian))
    initial_robot_state = [40, 0, 0.5 * np.pi]

    robot = Robot(
        initial_robot_state, field_of_view, robot_motion_noise, robot_measurement_noise
    )

    # Generate inputs and measurements
    num_steps = 80
    step_size = 3
    curviness = 0.5

    #robot_inputs = straight_robot_inputs(num_steps, step_size)
    robot_inputs = circular_robot_inputs(num_steps, step_size, 40)
    # robot_inputs = curved_robot_inputs(num_steps, step_size, curviness)
    # robot_inputs = random_robot_inputs(num_steps, step_size, curviness)

    # if you wanna use the dynamic landmarks
    # dynamic_ld_traj = generate_dynamic_ld_traj(dynamic_lds, num_steps, velocity_multiplier)

    # generate robot states and observations
    true_robot_states = [initial_robot_state]
    observations = []

    for movement, step in zip(robot_inputs, range(num_steps)):
        all_lds = static_lds

        # in case you want to include dynamic landmarks
        # all_lds = np.append(
        #     static_lds,
        #     dynamic_ld_traj[:, :3, step],
        #     axis=0)

        # process robot movement, set noise to false for the ground truth
        true_robot_states.append(robot.move(movement, noise=True))
        observations.append(robot.sense(all_lds))

    # use traj since there are no dynamic landmarks
    dummy_dynamic_ld_traj = None
    plotMap(static_lds, dummy_dynamic_ld_traj, true_robot_states, robot, map_size)

    # In[Estimation]

    # Initialize state matrices
    infinity = 1e6

    state_estimate = np.append(
        np.array([initial_robot_state]).T,
        np.zeros(
            (2 * (num_static_lds), 1)
        ),  # add + num_dynamic_lds for dynamic landmarks
        axis=0,
    )
    updated_state_estimate = state_estimate

    state_covariance = infinity * np.eye(2 * (num_static_lds) + 3)
    state_covariance[:3, :3] = np.zeros((3, 3))

    ld_classification_probabilities = 0.5 * np.ones((num_static_lds, 1))

    plotEstimate(state_estimate, state_covariance, robot, map_size)

    for movement, measurement in zip(robot_inputs, observations):

        updated_state_estimate, state_covariance = predict(
            updated_state_estimate, state_covariance, movement, robot_motion_noise
        )
        state_estimate = np.append(state_estimate, updated_state_estimate, axis=1)
        plotEstimate(state_estimate, state_covariance, robot, map_size)

        print("Measurements: {0:d}".format(len(measurement)))
        (
            updated_state_estimate,
            state_covariance,
            updated_classification_probabilities,
        ) = update(
            updated_state_estimate,
            state_covariance,
            measurement,
            ld_classification_probabilities[:, -1].reshape(
                num_static_lds, 1  # add + num_dynamic_lds for dynamic landmarks
            ),
            robot_measurement_noise,
        )
        state_estimate = np.append(state_estimate, updated_state_estimate, axis=1)

        ld_classification_probabilities = np.append(
            ld_classification_probabilities,
            updated_classification_probabilities,
            axis=1,
        )
        plotEstimate(state_estimate, state_covariance, robot, map_size)
        # this is useful for debugging but its slow
        #plotMeasurement(updated_state_estimate, state_covariance, measurement, num_static_lds)

        plotError(state_estimate, true_robot_states[: len(state_estimate[:, 0::2])][:])
        print("----------")
