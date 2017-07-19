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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 1000;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  particles.resize(num_particles);
  weights.resize(num_particles);

  default_random_engine gen;

  for (int indexOfParticle = 0; indexOfParticle < num_particles; ++indexOfParticle) {
    particles[indexOfParticle].id = indexOfParticle;
    particles[indexOfParticle].x = dist_x(gen);
    particles[indexOfParticle].y = dist_y(gen);
    particles[indexOfParticle].theta = dist_theta(gen);
    particles[indexOfParticle].weight = 1.0;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> noise_x(0, std_pos[0]);
  normal_distribution<double> noise_y(0, std_pos[1]);
  normal_distribution<double> noise_theta(0, std_pos[2]);
  default_random_engine gen;


  for (int indexOfParticle = 0; indexOfParticle < num_particles; ++indexOfParticle) {
    double x0 = particles[indexOfParticle].x;
    double y0 = particles[indexOfParticle].y;
    double theta0 = particles[indexOfParticle].theta;

    if (yaw_rate < 0.0001) {
      particles[indexOfParticle].x = x0 + velocity * delta_t * cos(theta0);
      particles[indexOfParticle].y = y0 + velocity * delta_t * sin(theta0);

      // Add noise
      particles[indexOfParticle].x += noise_x(gen);
      particles[indexOfParticle].y += noise_y(gen);
    }
    else {
      particles[indexOfParticle].x = x0 + (velocity / yaw_rate) * (sin(theta0 + yaw_rate * delta_t) - sin(theta0));
      particles[indexOfParticle].y = y0 + (velocity / yaw_rate) * (cos(theta0) - cos(theta0 + yaw_rate * delta_t));
      particles[indexOfParticle].theta = theta0 + yaw_rate * delta_t;

      // Add noise
      particles[indexOfParticle].x += noise_x(gen);
      particles[indexOfParticle].y += noise_y(gen);
      particles[indexOfParticle].theta += noise_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
  for (auto &observation : observations) {
    double min = std::numeric_limits<double>::max();

    for (auto &landmark : predicted) {
      double distance = dist(observation.x, observation.y, landmark.x, landmark.y);

      if(distance < min) {
        observation.id = landmark.id;
        min = distance;
      }
    }
  }
}

inline const double gaussian_2d(const LandmarkObs& obs, const Map::single_landmark_s &lm, const double sigma[])
{
  auto cov_x = sigma[0] * sigma[0];
  auto cov_y = sigma[1] * sigma[1];
  auto normalizer = 2.0 * M_PI * sigma[0] * sigma[1];
  auto dx = (obs.x - lm.x_f);
  auto dy = (obs.y - lm.y_f);
  return exp(-(dx * dx / (2 * cov_x) + dy * dy / (2 * cov_y))) / normalizer;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  for (auto &particle : particles) {
    // Transform observations from local coordinates to map coordinates with respect to the current particle
    std::vector<LandmarkObs> transformed_observations;
    for (auto &observation : observations) {
      LandmarkObs tobs;
      tobs.x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
      tobs.y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
      tobs.id = -1;

      transformed_observations.push_back(tobs);
    }

    // Collect plausible landmarks, only those which are within the sensor range
    std::vector<LandmarkObs> plausible_landmarks;
    for (auto &landmark : map_landmarks.landmark_list) {
      double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);

      if (distance <= sensor_range) {
        plausible_landmarks.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // Find closest landmark for each observation
    dataAssociation(plausible_landmarks, transformed_observations);

    // Update weights based on associated landmarks
    particle.weight = 1.0f;
    for (auto &observation : transformed_observations) {
      if (observation.id != -1) {
        Map::single_landmark_s landmark = map_landmarks.landmark_list[observation.id - 1];
        double prob_distribution = gaussian_2d(observation, landmark, std_landmark);
        particle.weight *= prob_distribution;
      }
    }

    // Save particles weight in separate array for easier resampling
    weights[particle.id] = particle.weight;
  }

  // Normalize weights
  double weight_sum = accumulate(weights.begin(), weights.end(), 0.0);

  for (auto &particle : particles) {
    particle.weight = particle.weight / weight_sum;
    weights[particle.id] = particle.weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::discrete_distribution<> sampling(weights.begin(), weights.end());
  default_random_engine gen;

  std::vector<Particle> resampled_particles;

  for (auto &particle : particles) {
    resampled_particles.push_back(particles[sampling(gen)]);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
