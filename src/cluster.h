/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./cluster.h
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#ifndef CLUSTER_H
#define CLUSTER_H

#include <iostream> 
#include <vector>
#include <list>
#include "segment.h"
#include "calculator.h"

using namespace std;

class Cluster {
   public:
      Cluster(int, int);
      static int counter;
      static int aval_id;
      // Update model parameters:
      // transition prob
      void init(const int, const int);
      void update_trans(vector<vector<float> >);
  void update_emission(vector<float>, int);
      void append_member(Segment*);
      void remove_members(Segment*);
      void set_cluster_id();
      void set_cluster_id(int);
      int get_member_num() const;
      int get_cluster_id() const;
      bool get_precompute_status() const {return precompute_status;}
      double compute_likelihood(const Segment&, const int);
      double compute_emission_likelihood(int, const float*, const int);
      vector<double> compute_posterior_weight(int, const float*, int);
      float get_state_trans_prob(int, int) const;
      int get_state_num() const { return state_num; }
      int get_dim() const { return vector_dim;}
      // list<Segment*>& get_members() { return members; }
      void show_member_len();
      void precompute(int, const float**);
      void set_precompute_status(const bool new_status);
  vector<float> get_emission(int index) {return emissions[index];} 
      void state_snapshot(const string&); 
      void update_age() {++age;}
      void increase_trans(const int, const int);
      void decrease_trans(const int, const int);
      void increase_cache(const int, const float*);
      void decrease_cache(const int, const float*);
      void set_trans(const float*);
      void set_member_num(const int s_member_num) {member_num = s_member_num;}
      int get_age() const {return age;}
      vector<vector<float> >& get_cache_trans() { return cache_trans;}
  const vector<float> get_cache_weights(const int);
      ~Cluster();
   private:
      // Store the cluster id
      int id;
      int age;
      int state_num;
      int vector_dim;
      int member_num;
      vector<vector<float> > trans;
      vector<vector<float> > cache_trans;
  vector<vector<float> > emissions; 
  vector<vector<float> > cache;
      // Store segments that belong to this cluster.
      // Utilities
      Calculator calculator;
      bool precompute_status;
};

#endif
