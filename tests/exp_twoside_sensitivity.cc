/**
 * @file exp_two_side_pos.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Sensitivity Experiment of Twoside Segment Graph
 * @date 2023-07-07
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_recursion_batch_reverse_prune.h"
#include "logger.h"
#include "range_index_base.h"
#include "reader.h"
#include "utils.h"

// #define LOG_DEBUG_MODE 1

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

long long before_memory, after_memory;

void execute_queries_in_range_search(
    BaseIndex &index, BaseIndex::SearchInfo &search_info,
    const DataWrapper &data_wrapper,
    const vector<int> &searchef_para_range_list) {
  timeval tt3, tt4;
  for (auto one_searchef : searchef_para_range_list) {
    gettimeofday(&tt3, NULL);
    for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
      int one_id = data_wrapper.query_ids.at(idx);
      BaseIndex::SearchParams s_params;
      s_params.query_K = data_wrapper.query_k;
      s_params.search_ef = one_searchef;
      s_params.control_batch_threshold = 1;
      s_params.query_range = data_wrapper.query_ranges.at(idx).second -
                             data_wrapper.query_ranges.at(idx).first;
      auto res = index.rangeFilteringSearchInRange(
          &s_params, &search_info, data_wrapper.querys.at(one_id),
          data_wrapper.query_ranges.at(idx));
      search_info.precision =
          countPrecision(data_wrapper.groundtruth.at(idx), res);
      search_info.approximate_ratio = countApproximationRatio(
          data_wrapper.nodes, data_wrapper.groundtruth.at(idx), res,
          data_wrapper.querys.at(one_id));
      search_info.RecordOneQuery(&s_params);
    }

    logTime(tt3, tt4, "total query time");
  }
}

void execute_queries_out_range_search(
    rangeindex::RecursionIndex &index, BaseIndex::SearchInfo &search_info,
    const DataWrapper &data_wrapper,
    const vector<int> &searchef_para_range_list) {
  timeval tt3, tt4;
  for (auto one_searchef : searchef_para_range_list) {
    gettimeofday(&tt3, NULL);
    for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++) {
      int one_id = data_wrapper.query_ids.at(idx);
      BaseIndex::SearchParams s_params;
      s_params.query_K = data_wrapper.query_k;
      s_params.search_ef = one_searchef;
      s_params.control_batch_threshold = 1;
      s_params.query_range = data_wrapper.query_ranges.at(idx).second -
                             data_wrapper.query_ranges.at(idx).first;
      auto res = index.rangeFilteringSearchOutBound(
          &s_params, &search_info, data_wrapper.querys.at(one_id),
          data_wrapper.query_ranges.at(idx));
      search_info.precision =
          countPrecision(data_wrapper.groundtruth.at(idx), res);
      search_info.approximate_ratio = countApproximationRatio(
          data_wrapper.nodes, data_wrapper.groundtruth.at(idx), res,
          data_wrapper.querys.at(one_id));
      search_info.RecordOneQuery(&s_params);
    }

    logTime(tt3, tt4, "total query time");
  }
}

int exp(string dataset, int data_size, string dataset_path, string method,
        string query_path, const bool is_amarel, const string gt_path) {
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif

  cout << "method: " << method << endl;
  int query_num = 500;
  vector<vector<float>> nodes;
  vector<int> search_keys;
  vector<vector<float>> querys;
  vector<int> querys_keys;

  if (dataset == "local") {
    cout << "Local test" << endl;
    query_num = 10;
  }

  int query_k = 10;
  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);

  data_wrapper.readData(dataset_path, query_path);

  vector<int> searchef_para_range_list = {50, 100, 200, 300, 400, 800};
  vector<int> index_k_list = {16, 32, 64};
  if (dataset == "local") {
    index_k_list = {8};
    searchef_para_range_list = {100, 400};
  }

  vector<int> ef_construction_list = {100, 200, 400, 800};
  vector<int> ef_max_list = {1000, 2000, 4000};
  string version = "sensitivity-position";
  if (is_amarel) {
    version = "amarel-" + version;
  }
  // string groundtruth_path = "../exp/groundtruth/groundtruth-twoside-sorted-"
  // +
  //                           version + "-" + data_wrapper.dataset + "-" +
  //                           std::to_string(data_wrapper.data_size) + ".csv";

  // data_wrapper.generateRangeFilteringQueriesAndGroundtruth(true,
  //                                                          groundtruth_path);

  data_wrapper.LoadGroundtruth(gt_path);

  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  delta_index_hnsw_full_reverse::L2Space ss(data_wrapper.data_dim);

  timeval t1, t2, t3, t4;

  // data_wrapper.version = version + "-fix-ef_max";
  // for (unsigned index_k : index_k_list) {
  //   for (unsigned ef_construction : ef_construction_list) {
  //     BaseIndex::IndexParams i_params;
  //     i_params.ef_max = 2000;
  //     i_params.ef_construction = ef_construction;
  //     i_params.K = index_k;
  //     i_params.ef_large_for_pruning = i_params.ef_construction;

  //     {
  //       cout << endl;
  //       i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
  //       rangeindex::RecursionIndex index(&ss, &data_wrapper);
  //       BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
  //                                         "MaxFull-OutBoundSearch",
  //                                         "twoside/search");
  //       cout << "Method: " << search_info.method << endl;
  //       cout << "parameters: ef_large_for_pruning and ef_max ( " +
  //                   to_string(i_params.ef_large_for_pruning) + "-" +
  //                   to_string(i_params.ef_max) + " )  index-k( "
  //            << i_params.K
  //            << " )  recursion_type ( " + to_string(i_params.recursion_type)
  //            +
  //                   " )"
  //            << endl;
  //       gettimeofday(&t1, NULL);
  //       index.buildIndex(&i_params);
  //       gettimeofday(&t2, NULL);
  //       logTime(t1, t2, "Build Index Time");
  //       cout << "Total # of Neighbors: " << index.index_info->nodes_amount
  //            << endl;

  //       execute_queries_out_range_search(index, search_info, data_wrapper,
  //                                        searchef_para_range_list);
  //     }

  //     {
  //       cout << endl;
  //       i_params.recursion_type = BaseIndex::IndexParams::MID_POS;
  //       rangeindex::RecursionIndex index(&ss, &data_wrapper);
  //       BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
  //                                         "MidFull-OutBoundSearch",
  //                                         "twoside/search");
  //       cout << "Method: " << search_info.method << endl;
  //       cout << "parameters: ef_large_for_pruning and ef_max ( " +
  //                   to_string(i_params.ef_large_for_pruning) + "-" +
  //                   to_string(i_params.ef_max) + " )  index-k( "
  //            << i_params.K
  //            << " )  recursion_type ( " + to_string(i_params.recursion_type)
  //            +
  //                   " )"
  //            << endl;
  //       gettimeofday(&t1, NULL);
  //       index.buildIndex(&i_params);
  //       gettimeofday(&t2, NULL);
  //       logTime(t1, t2, "Build Index Time");
  //       cout << "Total # of Neighbors: " << index.index_info->nodes_amount
  //            << endl;
  //       execute_queries_out_range_search(index, search_info, data_wrapper,
  //                                        searchef_para_range_list);
  //     }

  //     {
  //       cout << endl;
  //       i_params.recursion_type = BaseIndex::IndexParams::MIN_POS;
  //       rangeindex::RecursionIndex index(&ss, &data_wrapper);
  //       BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
  //                                         "MinFull-OutBoundSearch",
  //                                         "twoside/search");
  //       cout << "Method: " << search_info.method << endl;
  //       cout << "parameters: ef_large_for_pruning and ef_max ( " +
  //                   to_string(i_params.ef_large_for_pruning) + "-" +
  //                   to_string(i_params.ef_max) + " )  index-k( "
  //            << i_params.K
  //            << " )  recursion_type ( " + to_string(i_params.recursion_type)
  //            +
  //                   " )"
  //            << endl;
  //       gettimeofday(&t1, NULL);
  //       index.buildIndex(&i_params);
  //       gettimeofday(&t2, NULL);
  //       logTime(t1, t2, "Build Index Time");
  //       cout << "Total # of Neighbors: " << index.index_info->nodes_amount
  //            << endl;

  //       execute_queries_out_range_search(index, search_info, data_wrapper,
  //                                        searchef_para_range_list);
  //     }
  //   }
  // }

  // data_wrapper.version = version + "-hnsw";
  // for (unsigned index_k : index_k_list) {
  //   for (unsigned ef_construction : ef_construction_list) {
  //     BaseIndex::IndexParams i_params;
  //     i_params.ef_construction = ef_construction;
  //     i_params.K = index_k;
  //     i_params.ef_large_for_pruning = 0;
  //     i_params.ef_max = 0;
  //     {
  //       // Baseline
  //       cout << endl;
  //       BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
  //                                         "KnnFirstHnsw", "twoside/search");
  //       cout << "Method: " << search_info.method << endl;
  //       KnnFirstWrapper index(&data_wrapper);
  //       gettimeofday(&t1, NULL);
  //       index.buildIndex(&i_params);
  //       gettimeofday(&t2, NULL);
  //       logTime(t1, t2, "Build knnfirst HNSW Index Time");
  //       cout << "HNSW Total # of Neighbors: " <<
  //       index.index_info->nodes_amount
  //            << endl;
  //       execute_knn_first_search(index, search_info, data_wrapper,
  //                                searchef_para_range_list);
  //     }
  //   }
  // }

  data_wrapper.version = version + "-fix-ef_construction-2";
  for (unsigned index_k : index_k_list) {
    for (unsigned ef_max : ef_max_list) {
      BaseIndex::IndexParams i_params;
      i_params.ef_max = ef_max;
      i_params.ef_construction = 400;
      i_params.K = index_k;
      i_params.ef_large_for_pruning = i_params.ef_construction;

      {
        cout << endl;
        i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
        rangeindex::RecursionIndex index(&ss, &data_wrapper);
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                          "MaxFull-OutBoundSearch",
                                          "twoside/search");
        cout << "Method: " << search_info.method << endl;
        cout << "parameters: ef_large_for_pruning and ef_max ( " +
                    to_string(i_params.ef_large_for_pruning) + "-" +
                    to_string(i_params.ef_max) + " )  index-k( "
             << i_params.K
             << " )  recursion_type ( " + to_string(i_params.recursion_type) +
                    " )"
             << endl;
        gettimeofday(&t1, NULL);
        index.buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build Index Time");
        cout << "Total # of Neighbors: " << index.index_info->nodes_amount
             << endl;

        execute_queries_out_range_search(index, search_info, data_wrapper,
                                         searchef_para_range_list);
      }

      {
        cout << endl;
        i_params.recursion_type = BaseIndex::IndexParams::MID_POS;
        rangeindex::RecursionIndex index(&ss, &data_wrapper);
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                          "MidFull-OutBoundSearch",
                                          "twoside/search");
        cout << "Method: " << search_info.method << endl;
        cout << "parameters: ef_large_for_pruning and ef_max ( " +
                    to_string(i_params.ef_large_for_pruning) + "-" +
                    to_string(i_params.ef_max) + " )  index-k( "
             << i_params.K
             << " )  recursion_type ( " + to_string(i_params.recursion_type) +
                    " )"
             << endl;
        gettimeofday(&t1, NULL);
        index.buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build Index Time");
        cout << "Total # of Neighbors: " << index.index_info->nodes_amount
             << endl;
        execute_queries_out_range_search(index, search_info, data_wrapper,
                                         searchef_para_range_list);
      }

      {
        cout << endl;
        i_params.recursion_type = BaseIndex::IndexParams::MIN_POS;
        rangeindex::RecursionIndex index(&ss, &data_wrapper);
        BaseIndex::SearchInfo search_info(&data_wrapper, &i_params,
                                          "MinFull-OutBoundSearch",
                                          "twoside/search");
        cout << "Method: " << search_info.method << endl;
        cout << "parameters: ef_large_for_pruning and ef_max ( " +
                    to_string(i_params.ef_large_for_pruning) + "-" +
                    to_string(i_params.ef_max) + " )  index-k( "
             << i_params.K
             << " )  recursion_type ( " + to_string(i_params.recursion_type) +
                    " )"
             << endl;
        gettimeofday(&t1, NULL);
        index.buildIndex(&i_params);
        gettimeofday(&t2, NULL);
        logTime(t1, t2, "Build Index Time");
        cout << "Total # of Neighbors: " << index.index_info->nodes_amount
             << endl;

        execute_queries_out_range_search(index, search_info, data_wrapper,
                                         searchef_para_range_list);
      }
    }
  }

  return 0;
}

int main(int argc, char **argv) {
  string dataset = "biggraph";
  int data_size = 100000;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  bool is_amarel = false;
  string groundtruth_path = "";

  for (int i = 0; i < argc; i++) {
    string arg = argv[i];
    if (arg == "-dataset") dataset = string(argv[i + 1]);
    if (arg == "-N") data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path") dataset_path = string(argv[i + 1]);
    if (arg == "-method") method = string(argv[i + 1]);
    if (arg == "-query_path") query_path = string(argv[i + 1]);
    if (arg == "-amarel") is_amarel = true;
    if (arg == "-groundtruth_path") groundtruth_path = string(argv[i + 1]);
  }
  assert(groundtruth_path != "");

  exp(dataset, data_size, dataset_path, method, query_path, is_amarel,
      groundtruth_path);
  return 0;
}