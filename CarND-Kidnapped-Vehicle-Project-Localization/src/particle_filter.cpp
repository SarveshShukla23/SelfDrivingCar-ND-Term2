// Few sections of the code are taken from Lecture and Quizs.

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
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 20;
	default_random_engine gen;

	normal_distribution <double> n_x(x,std[0]);
	normal_distribution <double> n_y(y,std[1]);
	normal_distribution <double> n_theta(theta,std[2]);

	int i;
	for( i=0;i<num_particles;i++){
		Particle p;
		p.id = i;
		p.x = n_x(gen);
		p.y = n_y(gen);
		p.theta = n_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);

	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	int i;
	for(i = 0;i<num_particles;i++){

		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		double pred_x,pred_y,pred_theta;

		if (fabs(yaw_rate) < 0.0001) {
	    pred_x = p_x + velocity * cos(p_theta) * delta_t;
	    pred_y = p_y + velocity * sin(p_theta) * delta_t;
	    pred_theta = p_theta;
	  } else {
	    pred_x = p_x + (velocity/yaw_rate) * (sin(p_theta + (yaw_rate * delta_t)) - sin(p_theta));
	    pred_y = p_y + (velocity/yaw_rate) * (cos(p_theta) - cos(p_theta + (yaw_rate * delta_t)));
	    pred_theta = p_theta + (yaw_rate * delta_t);
	  }

		normal_distribution <double> n_x(pred_x,std_pos[0]);
		normal_distribution <double> n_y(pred_y,std_pos[1]);
		normal_distribution <double> n_theta(pred_theta,std_pos[2]);

		particles[i].x = n_x(gen);
		particles[i].y = n_y(gen);
		particles[i].theta = n_theta(gen);

	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations,double sensor_range) {

	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	int i,j;
	for(i=0;i<observations.size();i++){
		float low_dist = sensor_range * sqrt(2);
		int closest_id = -1;
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;
		for(j =0;j<predicted.size();j++){
			double pred_x = predicted[j].x;
		  double pred_y = predicted[j].y;
		  int pred_id = predicted[j].id;
		  double current_dist = dist(obs_x, obs_y, pred_x, pred_y);

		  if (current_dist < low_dist) {
		    low_dist = current_dist;
		    closest_id = pred_id;
		  }
		}
		observations[i].id = closest_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double w_norm = 0.0;
	int i,j,c,d;
	for(i = 0;i< num_particles;i++){
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;

		std::vector<LandmarkObs> t_obs;

		for(j = 0;j<observations.size();j++){
			LandmarkObs trans_obs;
			trans_obs.id = j;
			trans_obs.x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
			trans_obs.y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
			t_obs.push_back(trans_obs);
		}

		std::vector<LandmarkObs> p_landmarks;

		for (j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s c_landmark = map_landmarks.landmark_list[j];

			if ((fabs((p_x - c_landmark.x_f)) <= sensor_range) && (fabs((p_y - c_landmark.y_f)) <= sensor_range)) {
        p_landmarks.push_back(LandmarkObs {c_landmark.id_i, c_landmark.x_f, c_landmark.y_f});
    }
	}

  dataAssociation(p_landmarks,t_obs,sensor_range);

  particles[i].weight = 1.0;
	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];
	double sigma_x2 = pow(sigma_x,2);
	double sigma_y2 = pow(sigma_y,2);
	double norm = 1.0/(2.0 * M_PI * sigma_x*sigma_y);

	for(c =0;c<t_obs.size();c++){
		double t_obs_x = t_obs[c].x;
		double t_obs_y = t_obs[c].y;
		double t_obs_id = t_obs[c].id;
		double multi_prob = 1.0;

		for(d = 0;d<p_landmarks.size();d++){
			double p_landmark_x = p_landmarks[d].x;
			double p_landmark_y = p_landmarks[d].y;
			double p_landmark_id = p_landmarks[d].id;

			if (t_obs_id == p_landmark_id) {
          multi_prob = norm * exp(-1.0 * ((pow((t_obs_x - p_landmark_x), 2)/(2.0 * sigma_x2)) + (pow((t_obs_y - p_landmark_y), 2)/(2.0 * sigma_y2))));
          particles[i].weight *= multi_prob;
        }
		}
	}
	 w_norm += particles[i].weight;
}

for (int i = 0; i < particles.size(); i++) {
	 particles[i].weight /= w_norm;
	 weights[i] = particles[i].weight;
 }
}

void ParticleFilter::resample() {

	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	vector<Particle> resampled_particles;

	// Create a generator to be used for generating random particle index and beta value
	default_random_engine gen;

	//Generate random particle index
	uniform_int_distribution<int> particle_index(0, num_particles - 1);

	int current_index = particle_index(gen);

	double beta = 0.0;

	double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());

	for (int i = 0; i < particles.size(); i++) {
		uniform_real_distribution<double> random_weight(0.0, max_weight_2);
		beta += random_weight(gen);

	  while (beta > weights[current_index]) {
	    beta -= weights[current_index];
	    current_index = (current_index + 1) % num_particles;
	  }
	  resampled_particles.push_back(particles[current_index]);
	}
	particles = resampled_particles;


}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
