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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // set the number of particles
    num_particles = 1000;

    // Gaussian noise generator
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

    // Reset weights
    weights.clear();
    weights.resize(num_particles, 1);

    // Reset particles
    particles.clear();
    particles.resize(num_particles);

    for (auto &p: particles) {
        p.id     = 0;
        p.x      = dist_x(gen);
        p.y      = dist_y(gen);
        p.theta  = dist_t(gen);
        p.weight = 1;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (auto &p: particles) {

        if (yaw_rate > 0.0001) {
            double n_theta = p.theta + yaw_rate * delta_t;
            p.x = p.x + velocity / yaw_rate * (sin(n_theta) - sin(p.theta));
            p.y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(n_theta));
            p.theta = n_theta;
        }
        // small yaw rate
        else {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);
            p.theta = p.theta + yaw_rate * delta_t;
        }

        // add noise
        normal_distribution<double> dist_x(p.x,     std_pos[0]);
        normal_distribution<double> dist_y(p.y,     std_pos[1]);
        normal_distribution<double> dist_t(p.theta, std_pos[2]);

        p.x     = dist_x(gen);
        p.y     = dist_y(gen);
        p.theta = dist_t(gen);
    }
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

    std::vector<LandmarkObs> closest_obs;

    for (auto &pred: predicted) {
        double min_dist = numeric_limits<double>::max();
        LandmarkObs* closest = NULL;

        for (auto &obs: observations) {
            double distance = dist(pred.x, pred.y, obs.x, obs.y);
            if (distance < min_dist) {
                min_dist = distance;
                closest = &obs;
            }
        }

        closest_obs.push_back(*closest);
    }

    return closest_obs;
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
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

    for (int j = 0; j < particles.size(); ++j) {

        auto &p = particles[j];

        // extract in-range landmarks from map_landmarks
        vector<LandmarkObs> in_range;
        for (auto &landmark: map_landmarks.landmark_list) {
            if (dist(p.x, p.y, landmark.x_f, landmark.y_f) <= sensor_range) {
                LandmarkObs new_lm = { 0, landmark.x_f, landmark.y_f };
                in_range.push_back(new_lm);
            }
        }

        // transform observations by current particle
        vector<LandmarkObs> trans_obs;
        for (auto &o: observations) {
            LandmarkObs new_ob;
            new_ob.x = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
            new_ob.y = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;
            trans_obs.push_back(new_ob);
        }

        // data association
        vector<LandmarkObs> associated_obs = dataAssociation(in_range, trans_obs);

        // calculate bi-variate gaussian weight
        double total_weight = 1;
        for (int i = 0; i < in_range.size(); ++i) {

            auto landmark = in_range[i];
            auto observed = associated_obs[i];

            double mu_x = landmark.x;
            double mu_y = landmark.y;
            double x    = observed.x;
            double y    = observed.y;

            double denom = 2.0 * M_PI * std_landmark[0] * std_landmark[1];
            double d1 = pow(x - mu_x, 2) / pow(std_landmark[0], 2);
            double d2 = pow(y - mu_y, 2) / pow(std_landmark[1], 2);
            double numer = exp(-1 * d1 * d2 / 2);

            total_weight *= max(numer / denom, 0.001);
        }

        p.weight   = total_weight;
        weights[j] = total_weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // create distribution for weights
    discrete_distribution<int> dist_w(weights.begin(), weights.end());
    default_random_engine gen;

    vector<Particle> new_particles;
    for (int i = 0; i < num_particles; ++i) {
        new_particles.push_back(particles[dist_w(gen)]);
    }

    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
