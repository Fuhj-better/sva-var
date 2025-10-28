# CMake 最低版本要求
cmake_minimum_required(VERSION 3.20)

# 定义项目名称和 C++ 版本
project(RtlAnalyzerCpp CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# --- 找到并配置 slang 库 ---
set(slang_DIR "${CMAKE_SOURCE_DIR}/../../slang-out/lib/cmake/slang")
find_package(slang REQUIRED)

# --- 定义我们的可执行文件 ---
add_executable(
  analyzer
  src/main.cpp
  src/DependencyVisitor.cpp
)

# --- 【核心修复】显式添加 slang 的 include 目录 ---
# 我们手动告诉 'analyzer' 目标，去哪里寻找 #include "slang/..." 这样的头文件。
target_include_directories(analyzer PRIVATE
    "${CMAKE_SOURCE_DIR}/../../slang-out/include"
)
# ----------------------------------------------------

# 将我们的可执行文件与 slang 库链接
# 这部分依然是必需的，用于链接 .a 或 .so 库文件
target_link_libraries(analyzer PRIVATE slang::slang)