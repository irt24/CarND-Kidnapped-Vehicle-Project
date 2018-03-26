/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "helper_functions.h"
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  if (is_initialized) {
    return;
  }

  num_particles = 100;
  default_random_engine random_generator;
  normal_distribution<double> x_distribution(x, std[0]);
  normal_distribution<double> y_distribution(y, std[1]);
  normal_distribution<double> theta_distribution(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = x_distribution(random_generator);
    particle.y = y_distribution(random_generator);
    particle.theta = theta_distribution(random_generator);
    particle.weight = 1.0;
    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate) {
  default_random_engine gen;

  for (Particle& particle : particles) {
    // Make predictions based on the bicycle motion model.
    double predicted_x, predicted_y, predicted_theta;
    if (yaw_rate == 0) {
      predicted_x = particle.x + velocity * delta_t * cos(particle.theta);
      predicted_y = particle.y + velocity * delta_t * sin(particle.theta); 
      predicted_theta = particle.theta;
    } else {
      predicted_theta = particle.theta + yaw_rate * delta_t;
      predicted_x = particle.x + velocity / yaw_rate * (sin(predicted_theta) - sin(particle.theta));
      predicted_y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(predicted_theta));
    } 

    // Add Gaussian noise.
    normal_distribution<double> x_distribution(predicted_x, std_pos[0]);
    normal_distribution<double> y_distribution(predicted_y, std_pos[1]);
    normal_distribution<double> theta_distribution(predicted_theta, std_pos[2]);
    particle.x = x_distribution(gen);
    particle.y = y_distribution(gen);
    particle.theta = theta_distribution(gen);
  }
}

void ParticleFilter::updateWeights(
    double sensor_range,
    double std_landmark[], 
    const std::vector<LandmarkObs>& observations,
    Map& map) {
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  for (Particle& particle : particles) {
    // Prepare variables for debugging.
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    // Find the landmarks that are within this particle's sensor range.
    std::vector<Map::single_landmark_s> visible_landmarks;
    for (const auto& landmark : map.landmark_list) {
      if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
        visible_landmarks.push_back(landmark);
      }
    }

    double weight = 1.0;
    for (const LandmarkObs& obs_car : observations) {
      // Convert the observation from car coordinates to map coordinates.
      double obs_map_x =
          particle.x + obs_car.x * cos(particle.theta) - obs_car.y * sin(particle.theta);
      double obs_map_y =
          particle.y + obs_car.x * sin(particle.theta) + obs_car.y * cos(particle.theta);

      // Find the landmark that's most likely associated with this observation.
      double min_distance = numeric_limits<double>::infinity();
      Map::single_landmark_s* closest_landmark = nullptr;
      for (auto& landmark : visible_landmarks) {
        double distance = dist(landmark.x_f, landmark.y_f, obs_map_x, obs_map_y);
        if (distance < min_distance) {
          min_distance = distance;
          closest_landmark = &landmark; 
        }
      }

      // Update the weight of the particle based on how likely the observation is from this position.
      if (closest_landmark == nullptr) {
        weight = 0.0;
      } else {
        double dx = closest_landmark->x_f - obs_map_x;
        double dy = closest_landmark->y_f - obs_map_y;
        weight *= (1.0 / (2 * M_PI * std_x * std_y)
            * exp(-dx * dx / (2 * std_x * std_x))
            * exp(-dy * dy / (2 * std_y * std_y)));
        associations.push_back(closest_landmark->id_i);
        sense_x.push_back(obs_map_x);
        sense_y.push_back(obs_map_y);  
      }
    } 
    particle.weight = weight;
    SetAssociations(particle, associations, sense_x, sense_y);
  }  
}

void ParticleFilter::resample() {
  weights.clear();
  for (const Particle& particle : particles) {
    weights.push_back(particle.weight);
  }

  default_random_engine random_generator;
  discrete_distribution<int> particle_distribution(weights.begin(), weights.end());
  vector<Particle> resampled_particles;

  for (int i = 0; i < num_particles; i++) {
    int particle_index = particle_distribution(random_generator);
    resampled_particles.push_back(particles[particle_index]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(
    Particle& particle,
    const std::vector<int>& associations, 
    const std::vector<double>& sense_x,
    const std::vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
