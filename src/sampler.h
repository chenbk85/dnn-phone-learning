/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./sampler.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <mkl_vsl.h>
/*
#include <boost/random/uniform_real.hpp> // for normal_distribution.
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/linear_congruential.hpp>
*/
#include "segment.h"
#include "cluster.h"
#include "sample_boundary_info.h"
#include "calculator.h"
#include "storage.h"

using namespace std;
// using namespace boost;
// using namespace boost::random;

// typedef boost::mt19937 base_generator_type;

class Sampler {
   public:
      Sampler();
      static int seed;
      static float annealing;
      static int offset;
      // set up priors for the model
      void init_prior(const int, \
        const int, \
        const float, const float, \
        const float, const float, \
        const float, const float, \
        const float, const float);
      // sample the cluster for each segment
      SampleBoundInfo sample_h0_h1(Segment*, Segment*, Segment*, vector<Cluster*>&);
      void is_boundary(Segment*, Segment*, Segment*, list<Segment*>& , \
        vector<Cluster*>&, SampleBoundInfo&, vector<Bound*>::iterator);
      void is_not_boundary(Segment*, Segment*, Segment*, list<Segment*>& , \
        vector<Cluster*>&, SampleBoundInfo&, vector<Bound*>::iterator);
      void sample_more_than_cluster(Segment&, \
              vector<Cluster*>&, Cluster*);
      Cluster* sample_just_cluster(Segment&, vector<Cluster*>&);
      // sample cluster parameters
      void sample_hmm_parameters(Cluster&);
      int sample_index_from_log_distribution(vector<double>);
      int sample_index_from_distribution(vector<double>);
      bool decluster(Segment*, vector<Cluster*>&);
      bool clean_cluster(Segment*, vector<Cluster*>&);
      bool sample_boundary(Bound*);
      bool sample_boundary(vector<Bound*>::iterator, \
        list<Segment*>&, vector<Cluster*>&);
      void encluster(Segment&, vector<Cluster*>&, Cluster*);
      // sample from unit distribution
      float sample_from_unit();
      // sample from a diagonal covariance 
      const float* sample_from_gamma(int, const float*, \
        const float*, const float);
      float update_gamma_rate(float, float, float, float, float);
      // const float* sample_from_gamma_for_weight(const float*);
      const float* sample_from_gamma_for_multidim(const float*, int, float*);
      void sample_trans(vector<vector<float> >&, vector<vector<float> >&);
      Cluster* sample_cluster_from_base();
      // Cluster* sample_from_hash_for_cluster(Segment*, vector<Cluster*>&);
      // get DP prior
      double get_non_dp_prior(Cluster*) const;
      ~Sampler();
   private:
      int dim; 
      int state_num;
      float dp_alpha;
      float beta_alpha;
      float beta_beta;
      float gamma_shape;
      float gamma_weight_alpha;
      float gamma_trans_alpha;
      float norm_kappa;
      vector<double> boundary_prior;
      vector<double> boundary_prior_log;
      Calculator calculator;
      // base_generator_type generator;
      VSLStreamStatePtr stream; 
      Storage storage;
      int UNIT;
      int MEAN;
      int PRE;
      int EMIT;
};

#endif
