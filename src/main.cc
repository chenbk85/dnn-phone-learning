/* -*- C++ -*-
 *
 * Copyright (c) 2014
 * Spoken Language Systems Group
 * MIT Computer Science and Artificial Intelligence Laboratory
 * Massachusetts Institute of Technology
 *
 * All Rights Reserved

./main.cc
 *	FILE: cluster.cc 				                                *
 *										                            *
 *   				      				                            * 
 *   Chia-ying (Jackie) Lee <chiaying@csail.mit.edu>				*
 *   Feb 2014							                            *
*********************************************************************/
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdio.h>
#include <getopt.h>
#include <time.h>

#include "manager.h"

void print_usage() {
  cout << "Usage:" << endl;
   cout << "  ./dnn-phone-learning -d [data-list] -c [config-file] -g [gibbs-iter] -r [results-dir] -b [batch_size] -s [snapshot]" << endl;
}

string replaceChar(string str, char ch1, char ch2) {
  for (int i = 0; i < str.length(); ++i) {
    if (str[i] == ch1)
      str[i] = ch2;
  }

  return str;
}

int main(int argc, char* argv[]) {
  time_t timer = time(NULL);
  string result_dir = "results_" + replaceChar(ctime(&timer), ' ', '_');
   string data_list = "";
   string config_file = "";
   int gibbs_iter = 100;
   int batch_size = 100;
   string snapshot = "";

   int c;
   while ((c = getopt(argc, argv, "dcgrbs:")) != -1){
     if(c == 'd'){
       data_list = optarg;
     }
     else if(c == 'c'){
       config_file = optarg;
     }
     else if(c == 'g'){
       gibbs_iter = atoi(optarg);
     }
     else if(c == 'r'){
       result_dir = optarg;
     }
     else if(c == 'b'){
       batch_size = atoi(optarg);
     }
     else if(c == 's'){
       snapshot = optarg;
     }
     else{
       print_usage();
       return 1;
     }
   }

   Manager projectManager;
   if (!projectManager.load_config(config_file)) {
      cout << "Configuration file seems bad. Check " 
           << config_file << " to make sure." << endl;
      return -1;
   }
   else {
      cout << "Configuration file loaded successfully..." << endl;
   }

   projectManager.init_sampler();
   cout << "Sampler initialized successfully..." << endl;

   if (snapshot != "") {
      cout << "Loading snapshot..." << endl;
      if (!projectManager.load_snapshot(snapshot, data_list, batch_size)) {
         cout << "snapshot file seems bad. Check "
           << snapshot << "." << endl;
      }
      else {
         cout << "Snapshot loaded successfully..." << endl;
      }
   }
   else {
      if (!projectManager.load_bounds(data_list, batch_size)) {
         cout << "data list file seems bad. Check " 
            << data_list << " to make sure." << endl;
         return -1;
      }
      else {
         cout << "Data loaded successfully..." << endl;
      }
   }

   projectManager.gibbs_sampling(gibbs_iter, result_dir);

   return 0;
}
