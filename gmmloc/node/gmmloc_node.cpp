#include <ros/ros.h>

#include "gmmloc/gmmloc.h"

using namespace std;

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);

  ros::init(argc, argv, "gmmloc");

  ros::NodeHandle nh("~");

  gmmloc::GMMLoc system(nh);

  system.spin();

  // stop all threads
  system.stop();

  return 0;
}
