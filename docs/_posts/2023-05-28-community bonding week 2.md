---
layout: post
title: "Community Bonding: May 22 ~ May 28"
date:   2023-05-28 12:00:00
description: second week of the community bonding period; research data collection methods
tags: DataCollection CARLA AutonomousDriving
categories: WeeklyUpdates LiteratureResearch
---

This week marked an intense dive into literature and codebases to unravel the mechanisms and best practices of data collection. In Monday's meeting, we went over our findings from last week's literature review on state-of-the-art imitation learning algorithms for autonomous driving and set goals for this week. The focus was to comprehend and dissect the data collection methodologies utilized in various research papers. 

## Objectives
- [x] Detailed analysis of various research papers with a focus on their data collection methods.
- [ ] (Partially Completed) Explore the corresponding codebases to gain a practical understanding of the implementation.

## Findings
This week, I revisited the work done by Codevilla et al.[1], Hesham et al.[2], Chen et al.[3] and went over their codebases to understand how data collection was implemented.

### Overall
* **Expert demonstration**: Many works leverage CARLA's built-in autopilot function to gather expert demonstrations for imitation learning. However, it is also possible to use a human driver's input for a more realistic demonstration.
* **Simulation environment**: Usually in CARLA simulator, Town01 is used for training and Town02 is reserved for testing. 
* **Format**: In our case, each episode should consist of a sequence of tuples of (RGB image, semantic segmentation, measurements, control commands)
* **Data variability**: To improve the robustness of the learned policy, the data collected should cover various driving conditions including different weather conditions, traffic densities, and times of the day.
* **Data Augmentation**: This technique enriches the training set by simulating 'imperfect' scenarios and applying random transformations to each image during training, enhancing the model's ability to handle diverse situations.
* **Action Space**: Depending on the work, it can include the steering angle, the acceleration, the braking amount, or even high-level commands like 'turn left', 'turn right', 'go straight', etc.
    * **Route Planning**: If high-level commands are used to determine the direction of the vehicle at intersections, a navigator/planner could be used to derive the current high-level command for each frame. However, in our case, we would want to start small with routes that doesn't contain intersections.
* **Episode Duration**: Each episode should be long enough to contain meaningful driving behaviour but short enough to fit into the memory. It is common to set a maximum number of frames (e.g. 5,000) per episode.

### Data Augmentation
In my literature review, I found that many papers, including the two other papers listed, adopted the data augmentation practices introduced by Codevilla et al. These practices focus on two main aspects:

* Addressing the lack of 'imperfect' scenarios in the training data: As the training data typically lacks demonstrations of the agent recovering from non-ideal situations, it's necessary to augment the data with such instances. This can be achieved by injecting temporally correlated noise into the expert's driving commands during data collection. This simulates gradual deviation from the correct path and abrupt disturbances. However, only the expert driver's actual control commands are used in the training data.

* Applying random transformations to each image during training: These transformations include adjusting the image's contrast and brightness, adding Gaussian and salt-and-pepper noise, and introducing region dropout by masking a small section of the image black.

Note that some transformations such as tranlation or ratation should not be applied, as the control commands are not invariant of these transformations.

### Data Storage
Below are a few commonly-used data storage paradigms:
* **Images & json files**: RGB and semantic segmentation images can be stored as standard image files. JSON files can hold associated metadata for each frame such as well as the control commands and measurements. This format is human-readable and easily manipulated but may be slower to read/write in large quantities.
* **Pickle**: This is a Python-specific binary serialization format. Objects can be directly serialized and deserialized to and from byte streams, which makes storing complex objects convenient. However, it may not be suitable for very large datasets due to memory constraints.
* **LMDB**: This is a fast, memory-efficient database library. It can be used to store large amounts of data without loading the entire database into memory. This can greatly accelerate data retrieval and is especially useful for larger datasets.

### Useful Tools
There are some existing tools that can be mofidied to suit our needs or used as reference:
* **Carla Data Collector**(version 0.8.4): This tool allows the user to configure a dataset configuration file that contains a set of start/target positions, sensor settings, traffic settings etc. to generate a set of eepisodes from an expert demonstrator.
* **Carla Driving Benchmark**(version 0.8.4): This library provides a planner that uses a graph based approach and A* algorithm to find a path from the start location to the target location. It convieniently allows querying the high-level command at the current position.


### Next Steps
The plan is to incrementally increase the complexity of the scenarios from which we collect data. This approach ensures the fundamentals are in place before we dive into more advanced setups. Here's an outline:

* **Non-Intersection Lane Following with a Stationary Obstacle**: Begin with the simplest case: collect data where the agent doesn't encounter any intersections and only needs to follow the current lane, with one parked vehicle on its path. This will give us a feel for the data collection process, and help us understand the interaction between the agent and static objects in the environment.

* **Non-Intersection Lane Following with Dynamic Traffic**: Once we are comfortable with the basic data collection process, add more complexity by including moving cars on the map. This will give us a deeper understanding of how to handle dynamic objects in the environment.

* **Right Turn Only at Intersections**: With a solid understanding of data collection in simple non-intersection scenarios, we can then proceed to collect episodes where the agent only needs to turn right at intersections to arrive at the target location. This will bring into play the complexities of intersection navigation and multi-lane traffic.

* **Conditional Imitation Learning**: Finally, we can try to collect data in a way that matches the Conditional Imitation Learning paper, i.e., also record high-level commands like TURN_LEFT, TURN_RIGHT, GO_STRAIGHT, FOLLOW, etc. This will complete our imitation learning data collection procedure and lay a solid foundation for our following training and evaluation stages.



## References
[1] Codevilla, Felipe et al. “End-to-End Driving Via Conditional Imitation Learning.” 2018 IEEE International Conference on Robotics and Automation (ICRA) (2017): 1-9.

[2] Hesham M. Eraqi, Mohamed N. Moustafa, Jens Honer. Dynamic Conditional Imitation Learning for Autonomous Driving. IEEE Transactions on Intelligent Transportation Systems (ISSN: 1524-9050, Online ISSN: 1558-0016). Issue 12, Vol 23, Pages 22988-23001. DOI: 10.1109/TITS.2022.3214079, December 2022. [Impact Factor: 9.551]

[3] Chen, Dian et al. "Learning by Cheating." Conference on Robot Learning (CoRL). 2019.

[4] Carla Data Collector: https://github.com/carla-simulator/data-collector/tree/master

[5] Carla Driving Benchmark: https://github.com/carla-simulator/driving-benchmarks/tree/master

