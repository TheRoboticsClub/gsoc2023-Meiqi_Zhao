---
layout: page
title: roadmap
permalink: /roadmap/
nav: true
nav_order: 3
subtitle: <a href='#'>Affiliations</a>. Address. Contacts. Moto. Etc.
---

# Obstacle Avoidance for Autonomous Driving in CARLA Using Segmentation Deep Learning Models

## Abstract
Behavior Metrics[1] is an open-sourced autonomous driving network comparison tool that allows the user to load and test their autonomous driving models in different scenarios and compare the performance metrics against other models. Currently, Behavior Metrics only supports the follow-the-line task, where the vehicle must drive along a circuit while maintaining proximity to the center of the lane, and provides multiple trained models for benchmarking. This project aims to expand the current stack by adding support for a route navigation task where the agent follows a sequence of high-level commands to reach a destination while avoiding obstacles in CARLA simulator[2], as well as providing an end-to-end learning solution for the task. The ultimate goal is a model that enables an ego vehicle to follow the route while avoiding collision with dynamic objects, such as pedestrians and other vehicles, and comprehensive evaluation metrics for the new task.

## Project Overview
Here's a concise summary of the key milestones achieved during this GSoC program:

### Autonomous Driving Agent

In the initial phase, we successfully trained an autonomous driving agent that exhibited obstacle avoidance skills and could follow a predefined route while responding to high-level turning commands. This agent was built using imitation learning and deep neural networks.

### Expanding Behavior Metrics
During the latter part of the program, our focus shifted to expanding the capabilities of Behavior Metrics. We introduced support for generating traffic, allowing users to configure scenarios with other vehicles and pedestrians. Additionally, we added a new follow-route task, enabling agents to follow predefined routes involving various turns and junctions.

### New Evaluation Metrics
To accurately evaluate the performance of agents in these new scenarios, we designed and implemented new evaluation metrics. These metrics included route completion ratios, success rates, weighted success rates, and detailed infraction tracking, providing a comprehensive assessment of agent performance.

## Weekly Progress
### [Week 17](/gsoc2023-Meiqi_Zhao/blog/2023/week17)
In Week 17, the main focus was on expanding the evaluation metrics in Behavior Metrics to accommodate the new "follow-route" task. 

### [Week 16](/gsoc2023-Meiqi_Zhao/blog/2023/week16)
In Week 16, the primary focus was on developing Behavior Metrics. Specifically, the team worked on implementing the new "follow-route" task in Behavior Metrics. 

### [Week 15](/gsoc2023-Meiqi_Zhao/blog/2023/week15)
During Week 15, the focus was on improving model performance by experimenting with different architectures. Simultaneously, work continued on expanding Behavior Metrics by introducing user customization for different task types and integrating additional evaluation metrics.

### [Week 14](/gsoc2023-Meiqi_Zhao/blog/2023/week14)
During Week 14, the primary focus was on integrating the current model into Behavior Metrics to complete the project pipeline. The model was successfully added as a new brain in Behavior Metrics, enabling its use within the platform.

### [Week 12 & 13](/gsoc2023-Meiqi_Zhao/blog/2023/week12)
During Weeks 12 and 13, there was a focus on improving the model's obstacle avoidance capabilities. 

### [Week 11](/gsoc2023-Meiqi_Zhao/blog/2023/week11)
In Week 11, efforts to enhance the model's performance, particularly in obstacle avoidance, continued. Additionally, progress was made in integrating traffic generation functionality into the Behavior Metrics platform.

### [Week 10](/gsoc2023-Meiqi_Zhao/blog/2023/week10)
In Week 10, the focus shifted towards refining and evaluating the model's performance. Efforts also began to integrate the model into the Behavior Metrics platform, expanding its capabilities to include traffic.

### [Week 9](/gsoc2023-Meiqi_Zhao/blog/2023/week9)
In Week 9, we implemented Data Aggregation (DAgger) to iteratively enhance the model's behavior and refined the evaluation metrics to detect whether the model correctly follows turning instructions. 

### [Week 8](/gsoc2023-Meiqi_Zhao/blog/2023/week8)
In Week 8, we continued to address the "halting" problem encountered in the model, particularly when making turns at intersections. The team explored various strategies to optimize data collection, including data trimming and prioritization.

### [Week 7](/gsoc2023-Meiqi_Zhao/blog/2023/week7)
In Week 7, the focus was on enhancing the model's performance, particularly addressing the "halting" problem where the agent occasionally stops and gets stuck during navigation. 

### [Week 6](/gsoc2023-Meiqi_Zhao/blog/2023/week6)
In Week 6, the main focus was on enhancing the evaluation process by incorporating more sophisticated metrics inspired by the CARLA Leaderboard.

### [Week 5](/gsoc2023-Meiqi_Zhao/blog/2023/week5)
In Week 5, the primary focus was on refining the model's adherence to traffic lights by incorporating traffic light status as an additional input and experimenting with one-hot encoding for high-level commands. 

### [Week 4](/gsoc2023-Meiqi_Zhao/blog/2023/week4)
This week, efforts were focused on enhancing the model's ability to make turns at intersections in any direction by incorporating high-level commands. Data collection was adjusted to record these commands, and the model architecture was updated to accommodate them. 

### [Week 3](/gsoc2023-Meiqi_Zhao/blog/2023/week3)
This week, the focus was on improving the versatility of the lane-following model by enabling it to navigate routes with turns and intersections. 

### [Week 2](/gsoc2023-Meiqi_Zhao/blog/2023/week2)
In the second week of coding, the focus was on improving the training data quality. The primary enhancement was the introduction of noise injection into the expert agent's control commands to simulate recovery from disturbances. This approach proved effective in teaching the model how to auto-correct when it deviates from the center of the lane.

### [Week 1](/gsoc2023-Meiqi_Zhao/blog/2023/week1)
Implemented a data collection tool and collecting sample data, primarily consisting of simple scenarios with straight routes and dynamic obstacles. Simultaneously, a modified model, DeepestLSTMTinyPilotNet, was explored and trained on the collected data.

### [Community Bonding Week 2](http://127.0.0.1:4000/gsoc2023-Meiqi_Zhao/blog/2023/community-bonding-week-2/)
During the second week of community bonding, the focus was on researching data collection methods for the project. 

### [Community Bonding Week 1](http://127.0.0.1:4000/gsoc2023-Meiqi_Zhao/blog/2023/community-bonding-week-1/)
The week involved setting up the blog website, conducting literature research, and laying the project's groundwork. 

## References
[1] [https://github.com/JdeRobot/BehaviorMetrics](https://github.com/JdeRobot/BehaviorMetrics)

[2] *CARLA: An Open Urban Driving Simulator*,
Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun; PMLR 78:1-16
