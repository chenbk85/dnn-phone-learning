/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./cluster.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/

#include <iostream>
#include <fstream>
#include <cfloat>
#include <cmath>
#include "cluster.h"

using namespace std;

int Cluster::counter = 0;
int Cluster::aval_id = 0;

Cluster::Cluster(int s_num, int v_dim) {
// Cluster::Cluster() {
   state_num = s_num;
   vector_dim = v_dim;
   member_num = 0;
   // trans = new float[state_num];
   for (int i = 0; i < state_num; ++i) {
      vector<float> state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         state_trans.push_back(0.0);
      }
      trans.push_back(state_trans);
   }
   for (int i = 0; i < state_num; ++i) {
      vector<float> state_trans;
      for (int j = 0; j < state_num + 1; ++j) {
         state_trans.push_back(0.0);
      }
      cache_trans.push_back(state_trans);
   }
   id = -1;
   age = 0;
}

void Cluster::init(const int s_state_num, \
                   const int s_vector_dim) {
   state_num = s_state_num;
   vector_dim = s_vector_dim;
   for (int i = 0 ; i < state_num; ++i) {
     vector<float> new_weights(vector_dim); 
      emissions.push_back(new_weights);
   }
   for (int i = 0 ; i < state_num; ++i) {
      vector<float> inner_trans;
      for (int j = 0 ; j < state_num + 1; ++j) {
         inner_trans.push_back(0);
      }
      trans.push_back(inner_trans);
   }
   id = -1;
   age = 0;
}


void Cluster::increase_trans(const int i, const int j) {
   ++cache_trans[i][j];
}

void Cluster::decrease_trans(const int i, const int j) {
   --cache_trans[i][j];
}

void Cluster::update_emission(vector<float> weights, int index) {
   emissions[index] = weights;
}

void Cluster::update_trans(vector<vector<float> > new_trans) {
   trans = new_trans;
}

void Cluster::append_member(Segment* data) {
   int frame_num = data -> get_frame_num();
   for(int i = 0; i < frame_num - 1; ++i) {
      int s_t = data -> get_hidden_states(i);
      int s_t_1 = data -> get_hidden_states(i + 1);
      increase_trans(s_t, s_t_1);
   }
   // deal with the last frame
   int i = frame_num - 1;
   int s_t = data -> get_hidden_states(i);
   increase_trans(s_t, state_num);
   ++member_num;
}

void Cluster::remove_members(Segment* data) {
   int frame_num = data -> get_frame_num();
   for(int i = 0; i < frame_num - 1; ++i) {
      int s_t = data -> get_hidden_states(i);
      int s_t_1 = data -> get_hidden_states(i + 1);
      decrease_trans(s_t, s_t_1);
   }
   // deal with the last frame
   int i = frame_num - 1;
   int s_t = data -> get_hidden_states(i);
   decrease_trans(s_t, state_num);
   --member_num;
}

void Cluster::set_cluster_id(int s_id) {
   id = s_id;
}

void Cluster::set_cluster_id() {
   id = aval_id;
   aval_id++;
}

int Cluster::get_member_num() const {
   return member_num;
}

int Cluster::get_cluster_id() const {
   return id;
}

// Compute P(d|HMM)
double Cluster::compute_likelihood(const Segment& data, const int offset){
   double cur_scores[state_num];
   double pre_scores[state_num];
   for (int i = 0; i < state_num; ++i) {
      cur_scores[i] = 0.0;
      pre_scores[i] = 0.0;
   }
   
   // frame 0, state 0 (fixed)
   pre_scores[0] = compute_emission_likelihood(0, data.get_frame_i_likelihoods(0));
   if (data.get_frame_num() > 1) {
     // frame 1, all states (coming from state 0)
     for (int cur_state = 0; cur_state < state_num; ++cur_state) {
       double prob_emit = compute_emission_likelihood(cur_state, data.get_frame_i_likelihoods(1));
       cur_scores[cur_state] = pre_scores[0] + prob_emit + trans[0][curr_state]; 
     }

     // pre_scores are now frame 1
     for (int i = 0; i < state_num; ++i) {
       pre_scores[i] = cur_scores[i];
     }

      for (int i = 2; i < data.get_frame_num(); ++i) {
	// next frame, all states
	for (int cur_state = 0; cur_state < state_num; ++cur_state) {
	  for(int pre_state = 0; pre_state <= curr_state; pre_state++){
	    pre_scores[pre_state] += trans[pre_state][curr_state];
	  }
	  cur_scores[cur_state] = calculator.sum_logs(pre_scores, cur_state + 1);
	  double prob_emit_i = compute_emission_likelihood(cur_state, data.get_frame_i_likelihoods(i));
	  cur_scores[cur_state] += prob_emit_i;
	}
	// update pre_scores
	for (int j = 0; j < state_num; ++j ) {
	  pre_scores[j] = cur_scores[j];
	}
      }
   }
   // more than 1 frame has to end in last state
   if (data.get_frame_num() > 1) {
      return pre_scores[state_num - 1]; // + log(trans[state_num - 1][state_num]); 
   }
   else { // first frame is in first state
     return pre_scores[0]; // + log(trans[0][state_num]);
   }
}

// Compute P(x|Guassian) 
double Cluster::compute_emission_likelihood(int state, const float* likelihoods) {
  // likelihoods = P(state|data) = P(data|state)*P(state)/P(data)
  // P(data) is the same for all computations, so doesn't matter
  // assuming for now that P(state) also doesn't matter (i.e. is uniform), which gives P(state|data) ~ P(data|state)
  return likelihoods[counter*state + id]; //NOTE: this has to change with DP 
}

vector<vector<float> > Cluster::compute_forward_prob(Segment& data) {
  vector<vector<float> > forward_prob;

   // for the first frame, code directly                                                                                                                                                 
   // after the first frame, use the general pattern                                                                                                                                     
   vector<float> state_prob;
   state_prob.push_back(0.0);
   for (int s = 1; s < state_num; ++s) {
         state_prob.push_back(log(0));
   }
   forward_prob.push_back(state_prob);

   for (int i = 1; i < data.get_frame_num(); ++i) {
     state_prob.clear();
     for (int s = 0; s < state_num; ++s) {
       // Compute the forward prob for each state of each cluster                                                                                                                   
       if (s == 0) {
	 float trans_prob = forward_prob[0][i - 1] + trans[0][0];
	 float likelihood = compute_emission_likelihood(0, data.get_frame_i_data(i));
         state_prob.push_back(trans_prob + likelihood);
       }
       else {
	 vector<float> income;
	 for (int pre_state = 0; pre_state <= s; ++pre_state) {
	   income.push_back(forward_prob[pre_state][i - 1] + trans[pre_state][s]);
	 }
	 float trans_prob = calculator.sum_logs(income);
	 float likelihood = compute_emission_likelihood(s, data.get_frame_i_data(i));
	 state_prob.push_back(trans_prob + likelihood);
       }
     }
     forward_prob.push_back(state_prob);
   }
   return forward_prob;
}

vector<vector<float> > Cluster::compute_backward_prob(Segment& data) {
  vector<vector<float> > backward_prob;

   int last_frame_index = data.get_frame_num() - 1;
   vector<float> state_prob;
   if(last_frame_index == 0){ // only one frame
     vector<float> state_prob;
     state_prob.push_back(0.0);
     for (int s = 1; s < state_num; ++s) {
       state_prob.push_back(log(0));
     }
     backward_prob.push_back(state_prob);
   }
   else{
     for (int s = 0; s < state_num-1; ++s) {
       state_prob.push_back(log(0));
     }
     state_prob.push_back(0); // needs to end at last state

     int i = data.get_frame_num() - 2;
     for (; i >= 0; --i) {
       state_prob.clear();
       for (int s = 0; s < state_num; ++s) {
	 vector<float> outcome;
	 for (int next_state = s; next_state < state_num; ++next_state) {
	   outcome.push_back(backward_prob[next_state][0] + 
			     compute_emission_likelihood(next_state, data.get_frame_i_data(i + 1)) + 
			     trans[s][next_state]);
	 }
	 float intra_prob = calculator.sum_logs(outcome);
	 state_prob.push_back(intra_prob);
       }
       backward_prob.insert(backward_prob.begin(), 1, state_prob);
     }
   }
   return backward_prob;
}

vector<vector<float> > Cluster::compute_posterior(Segment& data) {
  vector<vector<float> > forward_prob = compute_forward_prob(data);
  vector<vector<float> > backward_prob = compute_backward_prob(data);

  vector<vector<float> > posteriors;
  for (int i = 0; i < data.get_frame_num(); ++i) {
    vector<float> frame_prob;
    frame_prob.clear();
    
    for (int s = 0; s < state_num; ++s) {
      float frame_forward = forward_prob[s][i];
      float frame_backward = backward_prob[s][i];
      frame_prob.push_back(frame_forward + frame_backward);
    }

    float normalizer = calculator.sum_logs(frame_prob);
    for (unsigned int l = 0; l < frame_prob.size(); ++l) {
      frame_prob[l] -= normalizer;
    }
    posteriors.push_back(frame_prob);
  }

  return posteriors;
}

void Cluster::run_vitterbi(Segment& data){
  vector<vector<float> > posteriors = compute_posterior(data);

  vector<vector<float> >  V ( data.get_frame_num(), vector<float> (state_num, 0.0));
  vector<vector<int> >  pathPointers ( data.get_frame_num(), vector<int> (state_num, 0));

  // initialize
  V[0][0] = 0;
  for(int s = 1; s < state_num; s++){
    V[0][s] = log(0);
  }
  pathPointers[0][0] = -1; // ????                                                                                                                                                  

  for (int i = 1; i < data.get_frame_num(); ++i) {
    // State 0 has to come from state 0
    V[i][0] = V[i-1][0] + posteriors[i][0];
    pathPointers[i][0] = 0;

    // other states can come from multiple places                                                                                                                                              
    for (unsigned int s = 1; s < state_num; ++s) {
      if(V[i-1][s] > V[i-1][s-1]){ // same state
	V[i][s] = V[i-1][s] + posteriors[i][s];
	pathPointers[i][s] = s;
      } else{ // previous state
	V[i][s] = V[i-1][s-1] + posteriors[i][s];
	pathPointers[i][s] = s-1;
      }
    }
  }
  int frame =  data.get_frame_num() - 1;
  int* new_hidden_states = new int[frame + 1];
  int state = state_num - 1;
  while(frame > 0){
    state = pathPointers[frame][state];
    frame--;
    new_hidden_states[frame] = state;
  }
  data.set_hidden_states(new_hidden_states);

}

void Cluster::set_trans(const float* s_trans) {
   for (int i = 0 ; i < state_num; ++i) {
      for (int j = 0 ; j < state_num + 1; ++j) {
         trans[i][j] = s_trans[i * (state_num + 1) + j];
      }
   }
}

float Cluster::get_state_trans_prob(int from, int to) const {
   return trans[from][to];
}

void Cluster::show_member_len() {
   cout << "I am Cluster " << id << 
     " and I have " << member_num << " members." << endl;
}
void Cluster::state_snapshot(const string& fn) {
   ofstream fout(fn.c_str(), ios::app);
   // write out members' info
   int member_len = member_num; 
   // write out member number
   fout.write(reinterpret_cast<char*> (&member_len), sizeof(int));
   // write state number
   fout.write(reinterpret_cast<char*> (&state_num), sizeof(int));
   // write vector_dim
   fout.write(reinterpret_cast<char*> (&vector_dim), sizeof(int));
   // write out trans info
   float copy_trans[state_num * (state_num + 1)];
   for (int i = 0; i < state_num; ++i) {
      for (int j = 0; j < state_num + 1; ++j) {
         copy_trans[i * (state_num + 1) + j] = trans[i][j]; 
      }
   }
   fout.write(reinterpret_cast<char*> (copy_trans), sizeof(float) * state_num * (state_num + 1)); 
   // write weights info
   for (int i = 0; i < state_num; ++i) {
     //vector<float> weights_i = emissions[i];
     //fout.write(reinterpret_cast<const char*> (weights_i), sizeof(float) * vector_dim);
   }
   // write out each member
   /*
   vector<Segment*>::iterator iter_segments;
   for(iter_segments = members.begin(); iter_segments != \
     members.end(); ++iter_segments) {
      // write tag
      string tag = (*iter_segments) -> get_tag();
      int tag_len = tag.length() + 1;
      // write tag length
      fout.write(reinterpret_cast<char*> (&tag_len), sizeof(int));
      fout.write(reinterpret_cast<const char*> (tag.c_str()), tag_len);
      // write start frame
      int start = (*iter_segments) -> get_start_frame();
      fout.write(reinterpret_cast<char*> (&start), sizeof(int));
      // write end frame
      int end = (*iter_segments) -> get_end_frame();
      fout.write(reinterpret_cast<char*> (&end), sizeof(int));
      // write cluster id
      int cluster_id = (*iter_segments) -> get_cluster_id();
      fout.write(reinterpret_cast<char*> (&cluster_id), sizeof(int));
      // write frame number
      int frame_num = (*iter_segments) -> get_frame_num();
      fout.write(reinterpret_cast<char*> (&frame_num), sizeof(int));
      // write hidden states
      const int* hidden_states = (*iter_segments) -> get_hidden_states_all();
      fout.write(reinterpret_cast<const char*> (hidden_states), \
        sizeof(int) * frame_num);
      // write mixture id
      const int* mixture_id = (*iter_segments) -> get_mixture_id_all();
      fout.write(reinterpret_cast<const char*> (mixture_id), \
        sizeof(int) * frame_num);
      for (int i = 0 ; i < frame_num; ++i) {
         const float* frame_i = (*iter_segments) -> get_frame_i_data(i);
         fout.write(reinterpret_cast<const char*> (frame_i), \
           sizeof(float) * vector_dim);
      }
   }
   */
   fout.close();
}

Cluster::~Cluster(){
   /*
   if (id != -1) {
      cout << "cluster " << id << " has been destructed" << endl;
   }
   */
}

