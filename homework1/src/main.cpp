#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <ctime>
#include <vector>
#include <fstream>

namespace {

const char* TIME_LOG_FILE_NAME = "time_log";

inline double Function(double arg) {
  return 4 / (1 + arg * arg);
}

double ComputeIntegral(double start, double end, size_t section_number) {
  double result = 0;
  double section_size = (end - start) / section_number;
  for (size_t i = 0; i < section_number; ++i) {
    double position = start + i * section_size;
    result += Function(position) + Function(position + section_size);
  }
  return result * section_size / 2;
}

void LogTime(size_t proc_num, size_t sections_num, double time_ms) {
  std::ofstream fl;
  fl.open(TIME_LOG_FILE_NAME);
  fl << proc_num << " " << sections_num << " " << time_ms << "\n";
  fl.close();

  std::cout << std::setprecision(5) << "\nTime: "
            << time_ms << " ms" << std::endl;
}

void Log() {
  std::cout << std::endl;
}

template <typename Head, typename... Rest>
void Log(Head head, Rest... rest) {
  std::cout << head << " ";
  Log(rest...);
}

void LogPartResults(const std::vector<double>& part_results) {
  for (size_t i = 0; i < part_results.size(); ++i) {
      Log("Process", i, "part", part_results[i]);
    }
}

} // namespace

int main(int argc, char** argv) {
  int section_number = atoi(argv[1]);

  double result = 0;
  MPI_Init(&argc, &argv);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Status status;
  double borders[2];

  double time_start;
  if (world_rank == 0) {
    time_start = MPI_Wtime();
    for (size_t i = 1; i < world_size; ++i) {
      borders[0] = static_cast<double>(i) / world_size;
      borders[1] = static_cast<double>(i + 1) / world_size;
      MPI_Send(borders, 2 , MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
    borders[0] = 0;
    borders[1] = 1. / world_size;
  } else {
    MPI_Recv(borders, 2 , MPI_DOUBLE, MPI_ANY_SOURCE, 0,
             MPI_COMM_WORLD, &status);
  }

  int part_section_number = world_rank != world_size - 1
      ? section_number / world_size
      : section_number - (section_number / world_size) * (world_size - 1);

  double integral_part =
      ComputeIntegral(borders[0], borders[1], part_section_number);

  if (world_rank == 0) {
    double part;
    result += integral_part;

    std::vector<double> part_results;
    part_results.push_back(integral_part);
    for (size_t i = 1; i < world_size; ++i) {
      MPI_Recv(&part, 1 , MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
      result += part;
      part_results.push_back(part);
    }

    double time_end = MPI_Wtime();
    double time_ms = (time_end - time_start) * 1000;

    LogTime(world_size, section_number, time_ms);
    Log("Result:", result);
    LogPartResults(part_results);
    Log("First computation:", ComputeIntegral(0, 1, section_number));

  } else {
    MPI_Send(&integral_part, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}