cmake_minimum_required(VERSION 3.8)
project(helipad_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/helipad_detector_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/HelipadDetector.py
  scripts/Helipad_3D.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  cfg
  model
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
