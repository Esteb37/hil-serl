cmake_minimum_required(VERSION 3.8)
project(spot_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BatteryState.msg"
  "msg/BatteryStateArray.msg"
  "msg/Lease.msg"
  "msg/LeaseArray.msg"
  "msg/LeaseOwner.msg"
  "msg/LeaseResource.msg"
  DEPENDENCIES builtin_interfaces std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
