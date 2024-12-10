/**
 * @file exp_halfbound.cc
 * @author Chaoji Zuo (chaoji.zuo@rutgers.edu)
 * @brief Benchmark Half-Bounded Range Filter Search
 * @date 2023-12-22
 *
 * @copyright Copyright (c) 2023
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include <iomanip>
// #include "baselines/knn_first_hnsw.h"
#include "data_processing.h"
#include "data_wrapper.h"
#include "index_base.h"
#include "logger.h"
#include "reader.h"
#include "segment_graph_2d.h"
#include "utils.h"

#ifdef __linux__
#include "sys/sysinfo.h"
#include "sys/types.h"
#endif

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
void write_to_csv(const DataWrapper &data_wrapper,
                  const std::vector<int> &search_ef_list,
                  const std::map<int, std::map<int, std::pair<float, float>>> &result_recorder,
                  const std::map<int, std::map<int, float>> &comparison_recorder,
                  const std::map<int, std::map<int, float>> &hop_recorder);
void log_result_recorder(
    const std::map<int, std::pair<float, float>> &result_recorder,
    const std::map<int, float> &comparison_recorder, const std::map<int, float> &hop_recorder, const int amount)
{
  for (auto it : result_recorder)
  {
    cout << std::setiosflags(ios::fixed) << std::setprecision(4)
         << "range: " << it.first
         << "\t recall: " << it.second.first / (amount / result_recorder.size())
         << "\t QPS: " << std::setprecision(0)
         << (amount / result_recorder.size()) / it.second.second << "\t Comps: "
         << comparison_recorder.at(it.first) / (amount / result_recorder.size()) << "\t Hops: "
         << hop_recorder.at(it.first) / (amount / result_recorder.size())
         << endl;
  }
}

int main(int argc, char **argv)
{
#ifdef USE_SSE
  cout << "Use SSE" << endl;
#endif
#ifdef USE_AVX
  cout << "Use AVX" << endl;
#endif
#ifdef USE_AVX512
  cout << "Use AVX512" << endl;
#endif
#ifndef NO_PARALLEL_BUILD
  cout << "Index Construct Parallelly" << endl;
#endif

  // Parameters
  string dataset = "deep";
  int data_size = 1000000;
  string dataset_path = "";
  string method = "";
  string query_path = "";
  string range_path = "/root/data/ranges/intrng10k_1000000";
  string groundtruth_path = "/root/data/gt/gt_sift_k10";
  vector<int> index_k_list = {8};
  vector<int> ef_construction_list = {100};
  int query_num = 1000;
  int query_k = 10;
  vector<int> ef_max_list = {300};

  string indexk_str = "";
  string ef_con_str = "";
  // string ef_max_str = "";
  string version = "Benchmark";

  for (int i = 0; i < argc; i++)
  {
    string arg = argv[i];
    if (arg == "-dataset")
      dataset = string(argv[i + 1]);
    if (arg == "-N")
      data_size = atoi(argv[i + 1]);
    if (arg == "-dataset_path")
      dataset_path = string(argv[i + 1]);
    if (arg == "-query_path")
      query_path = string(argv[i + 1]);
    if (arg == "-groundtruth_path")
      groundtruth_path = string(argv[i + 1]);
    // if (arg == "-ef_max") ef_max_str = string(argv[i + 1]);
    if (arg == "-method")
      method = string(argv[i + 1]);
  }

  assert(index_k_list.size() != 0);
  assert(ef_construction_list.size() != 0);
  // assert(groundtruth_path != "");

  DataWrapper data_wrapper(query_num, query_k, dataset, data_size);
  data_wrapper.readData(dataset_path, query_path);
  data_wrapper.frac_num = 18;
  // Generate groundtruth
  // data_wrapper.generateRangeFilteringQueriesAndGroundtruthBenchmark(false);
  // Or you can load groundtruth from the given path
  data_wrapper.LoadGroundtruth(range_path, groundtruth_path);
  cout << "data_wrapper.groundtruth.size: " << data_wrapper.groundtruth.size() << endl;
  cout << "data_wrapper.query_ranges.size: " << data_wrapper.query_ranges.size() << endl;
  cout << "data_wrapper.query_fracs.size: " << data_wrapper.query_fracs.size() << endl;
  cout << "data_wrapper.query_ids.size: " << data_wrapper.query_ids.size() << endl;
  cout << "data_wrapper.querys.size: " << data_wrapper.querys.size() << endl;
  assert(data_wrapper.query_ids.size() == data_wrapper.query_ranges.size());

  // vector<int> searchef_para_range_list = {16, 64, 256};
  // vector<int> searchef_para_range_list = {700, 500, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60};
  vector<int> searchef_para_range_list = {1700, 1500, 1300, 1100, 1000, 900, 800, 700, 600, 500, 400, 300, 250, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10};

  cout << "index K:" << endl;
  print_set(index_k_list);
  cout << "ef construction:" << endl;
  print_set(ef_construction_list);
  cout << "search ef:" << endl;
  print_set(searchef_para_range_list);

  data_wrapper.version = version;

  base_hnsw::L2Space ss(data_wrapper.data_dim);

  timeval t1, t2;

  for (unsigned index_k : index_k_list)
  {
    for (unsigned ef_max : ef_max_list)
    {
      for (unsigned ef_construction : ef_construction_list)
      {
        BaseIndex::IndexParams i_params(index_k, ef_construction,
                                        ef_construction, ef_max);
        {
          cout << endl;
          i_params.recursion_type = BaseIndex::IndexParams::MAX_POS;
          SeRF::IndexSegmentGraph2D index(&ss, &data_wrapper);
          // rangeindex::RecursionIndex index(&ss, &data_wrapper);
          BaseIndex::SearchInfo search_info(&data_wrapper, &i_params, "SeRF_2D",
                                            "benchmark");

          cout << "Method: " << search_info.method << endl;
          cout << "parameters: ef_construction ( " +
                      to_string(i_params.ef_construction) + " )  index-k( "
               << i_params.K << ")  ef_max (" << i_params.ef_max << ") "
               << endl;
          gettimeofday(&t1, NULL);
          index.buildIndex(&i_params);
          gettimeofday(&t2, NULL);
          logTime(t1, t2, "Build Index Time");
          cout << "Total # of Neighbors: " << index.index_info->nodes_amount
               << endl;

          {
            timeval tt3, tt4;
            BaseIndex::SearchParams s_params;
            s_params.query_K = data_wrapper.query_k;
            std::map<int, std::map<int, std::pair<float, float>>>
                result_recorder; // first->precision, second->query_time
            std::map<int, std::map<int, float>> comparison_recorder;
            std::map<int, std::map<int, float>> hop_recorder;
            for (auto one_searchef : searchef_para_range_list)
            {
              s_params.search_ef = one_searchef;

              gettimeofday(&tt3, NULL);
              for (int idx = 0; idx < data_wrapper.query_ids.size(); idx++)
              {
                // cout << data_wrapper.query_fracs.at(idx) << ",";
                int one_id = data_wrapper.query_ids.at(idx);
                s_params.query_range =
                    data_wrapper.query_ranges.at(idx).second -
                    data_wrapper.query_ranges.at(idx).first + 1;
                auto res = index.rangeFilteringSearchOutBound(
                    &s_params, &search_info, data_wrapper.querys.at(one_id),
                    data_wrapper.query_ranges.at(idx));
                search_info.precision =
                    countPrecision(data_wrapper.groundtruth.at(idx), res);
                // cout << "groundtruth at idx " << idx << ": ";
                // print_set(data_wrapper.groundtruth.at(idx));
                // cout << "res: ";
                // print_set(res);
                // cout << "search_info.precision: " << search_info.precision << endl;
                result_recorder[data_wrapper.query_fracs.at(idx)][s_params.search_ef].first +=
                    search_info.precision;
                result_recorder[data_wrapper.query_fracs.at(idx)][s_params.search_ef].second +=
                    search_info.internal_search_time;
                comparison_recorder[data_wrapper.query_fracs.at(idx)][s_params.search_ef] +=
                    search_info.total_comparison;
                hop_recorder[data_wrapper.query_fracs.at(idx)][s_params.search_ef] +=
                    search_info.total_hop;
              }

              cout << endl
                   << "Search ef: " << one_searchef << endl;
              //  << "========================" << endl;
              // log_result_recorder(result_recorder, comparison_recorder, hop_recorder,
              //                     data_wrapper.query_ids.size());
              // cout << "========================" << endl;
              logTime(tt3, tt4, "total query time");
            }
            write_to_csv(data_wrapper, searchef_para_range_list, result_recorder, comparison_recorder, hop_recorder);
          }
        }
      }
    }
  }

  return 0;
}

void write_to_csv(const DataWrapper &data_wrapper,
                  const std::vector<int> &search_ef_list,
                  const std::map<int, std::map<int, std::pair<float, float>>> &result_recorder,
                  const std::map<int, std::map<int, float>> &comparison_recorder,
                  const std::map<int, std::map<int, float>> &hop_recorder)
{
  for (const auto &range_entry : result_recorder)
  {
    int range = range_entry.first; // 当前range
    std::string filename = "/root/SeRF/exp/result/" + std::to_string(range) + ".csv";

    std::ofstream file(filename);
    // file << "search_ef,fa,fb,fc,fd\n";

    for (auto search_ef : search_ef_list)
    {
      auto result_it = range_entry.second.find(search_ef);
      auto comparison_it = comparison_recorder.at(range).find(search_ef);
      auto hop_it = hop_recorder.at(range).find(search_ef);

      if (result_it != range_entry.second.end() &&
          comparison_it != comparison_recorder.at(range).end() &&
          hop_it != hop_recorder.at(range).end())
      {
        // cout << "data_wrapper.query_ids.size(): " << data_wrapper.query_ids.size() << endl;
        // cout << "data_wrapper.frac_num: " << data_wrapper.frac_num << endl;
        float recall = (result_it->second.first) / (float)(data_wrapper.query_ids.size() / (data_wrapper.frac_num));
        float qps = (data_wrapper.query_ids.size() / (data_wrapper.frac_num)) / (result_it->second.second);
        float comp = (comparison_it->second) / (float)(data_wrapper.query_ids.size() / (data_wrapper.frac_num));
        float hop = (hop_it->second) / (float)(data_wrapper.query_ids.size() / (data_wrapper.frac_num));

        file << search_ef << "," << recall << "," << qps << "," << comp << "," << hop << "\n";
      }
    }

    file.close();
    std::cout << "Generated CSV for range " << range << " at " << filename << std::endl;
  }
}