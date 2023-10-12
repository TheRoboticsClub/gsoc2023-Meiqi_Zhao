---
layout: post
title: "Community Bonding: May 15 ~ May 21"
date:   2023-05-20 11:05:00
description: first week of the community bonding period; literature search on imitation learning and autonomous driving
tags: JdeRobot BehaviorMetrics GSoC ImitationLearning AutonomousDriving
categories: WeeklyUpdates LiteratureResearch
---

I'm thrilled to be part of Google Summer of Code 2023! This week marks the official start of the community bonding period at JdeRobot, and I couldn't be more excited. My project, "Obstacle Avoidance for Autonomous Driving in CARLA Using Segmentation Deep Learning Models", promises to be a fascinating, research-oriented endeavor, which will see me dive deep into the fields of imitation learning and autonomous driving. Let's get started!

This week we held our first official meeting, where we introduced ourselves, and the mentors went over the program's logistics. We also laid out the goals for the week, which are listed below.

## Preliminaries
Before the official start of GSoC, I had already familiarized myself with the Behavior Metrics project's codebase. My initial involvement came during the application phase, where I actively engaged with the project, contributing to its development by opening and successfully completing two GitHub issues and submitted pull requests ([#606](https://github.com/JdeRobot/BehaviorMetrics/pull/606), [#620](https://github.com/JdeRobot/BehaviorMetrics/pull/620)) that added obstacles to the autonomous driving task in CARLA simulator. Before, Behavior Metrics only supported scenarios where the autonomous vehicle simply followed the lane without encountering obstacles. The goal for this summer is to introduce an obstacle avoidance task, training and testing the autonomous vehicle in scenarios populated with other vehicles and pedestrians. Currently, only one dynamic obstacle is added for experimental purposes. More complex simulation senarios will be designed and implemented down the road. This prelimiary experience with the project has provided me with a strong foundation to start my GSoC journey.

## Objectives
- [x] Set up blog website using Jekyll and Github Pages
- [x] Set up the roadmap page of the blog
- [x] Conduct literature research on imitation learning, autonomous driving, and obstacle avoidance
- [x] Write a first blog documenting this week's progress

## Progress

This week I've focused on setting up this blog website. It's crucial to have a centralized platform to document my journey and update my progress throughout the project. As suggested by the mentors, I decided to use Jekyll, a static site generator, along with GitHub Pages for hosting the website.

During my literature search, I encountered an interesting paper by Zhou et al[1]. which compares an end-to-end pixels-to-actions baseline model with models that receives computer vision representations as additional modalities, including depth images, semantic and instance segmentation masks, optical flow, and albedo. The experimental results indicated that ground truth vision representations significantly improved visuomotor policies in an urban driving task in terms of success rate and weighted success rate. Furthermore, even imperfect representations predicted by a simple U-Net provided some advantages. In our project, we will begin with ground truth semantic segmentation provided by the CARLA simulator. If time permits, it might be interesting to compare the performance of the model with and without the segmentation mask.

Another notable paper by Eraqi et al.[2] proposed a conditional imitation model improved by feature-level fusion of lidar scan with RGB camera image. The idea is that the camera and lidar are complementary, as the camera perceives color and texture of objects while lidar captures depth information and is less sensitive to ambient light. In our project, we will introduce semantic segmentation as an additional input to the model. Therefore, experimenting with different fusion strategies could be interesting.

For our project, we might consider adopting the data collection method used by Zhou et al., which involves only sampling routes where target locations are reachable by turning right at every intersection from the starting locations. This approach effectively sidesteps the need to implement a global route planner, as used in Eraqi et al.

## References
[1] Brady Zhou, et al. "Does computer vision matter for action?". CoRR abs/1905.12887. (2019).

[2] Hesham M. Eraqi, et al. "Dynamic Conditional Imitation Learning for Autonomous Driving". IEEE Transactions on Intelligent Transportation Systems 23. 12(2022): 22988–23001.

