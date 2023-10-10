import numpy as np

COLLISION_WALKER_PENALTY = 0.5
COLLISION_VEHICLE_PENALTY = 0.6
COLLISION_OTHER_PENALTY = 0.65
RED_LIGHT_PENALTY = 0.7
TIMEOUT_PENALTY = 0.7
WRONG_TURN_PENALTY = 0.7


def calculate_stats(distances, success_record):
    assert len(distances) == len(success_record), "Mismatch between lengths of distances and success_record arrays"
    
    num_episodes = len(success_record)
    
    # success rate
    success_rate = sum(success_record) / num_episodes
    
    # sccess rate weighted by track length
    weighted_success_rate = sum([dist * success for dist, success in zip(distances, success_record)]) / sum(distances)
    
    # Average distance traveled for failed episodes only
    fail_distances = [dist for dist, success in zip(distances, success_record) if success == 0]
    average_fail_distance = sum(fail_distances) / len(fail_distances) if fail_distances else 0
    
    return success_rate, weighted_success_rate, average_fail_distance

class MetricsRecorder:
    def __init__(self):
        self.info = []
        self.success_record = []
        self.distances = []
        self.driving_scores = []
        self.total_infractions = {'red_light': 0,
                        'nonjunction_time_out': 0,
                        'junction_time_out': 0,
                        'collision_vehicle': 0,
                        'collision_walker': 0,
                        'collision_other': 0,
                        'turning': 0}
        self.infractions = None
        self.route_completion = 0.0
    
    def start_episode(self):
        self.infractions = {'red_light': 0,
                       'nonjunction_time_out': 0,
                       'junction_time_out': 0,
                       'collision_vehicle': 0,
                       'collision_walker': 0,
                       'collision_other': 0,
                       'turning': 0}
    
    def record(self, name):
        self.infractions[name] += 1

    def end_episode(self):
        infraction_penalty = COLLISION_VEHICLE_PENALTY ** self.infractions['collision_walker'] * \
                            COLLISION_WALKER_PENALTY ** self.infractions['collision_vehicle'] * \
                            COLLISION_OTHER_PENALTY ** self.infractions['collision_other'] * \
                            RED_LIGHT_PENALTY ** self.infractions['red_light'] * \
                            TIMEOUT_PENALTY ** (self.infractions['junction_time_out'] + self.infractions['nonjunction_time_out']) * \
                            WRONG_TURN_PENALTY ** self.infractions['turning']
        driving_score = self.route_completion * infraction_penalty
        for infraction_type in ['collision_walker', 'collision_vehicle', 'collision_other', 'red_light', 'nonjunction_time_out', 'junction_time_out', 'turning']:
            self.total_infractions[infraction_type] += self.infractions[infraction_type]
        self.info.append(f'route compleition: {self.route_completion}')
        self.info.append(f'infraction penalty: {infraction_penalty}')
        self.info.append(f'driving score: {driving_score}\n')
        self.driving_scores.append(driving_score)
    
    def get_evaluation_results(self):
        success_rate, weighted_success_rate, average_fail_distance = calculate_stats(np.array(self.distances), np.array(self.success_record))
        self.info.append(f'success rate: {success_rate}')
        self.info.append(f'success rate weighted by track length: {weighted_success_rate}')
        self.info.append(f'average distance traveled in failed cases: {average_fail_distance}')
        self.info.append(f'average driving score: {np.mean(self.driving_scores)}')
        self.info.append(f'infractions: {self.total_infractions}')
        print('------------RESULTS-------------')
        for line in self.info: print(line)
        print('------------END-------------')