# 1. 确保您在 analyzer_cpp 目录下
cd /data/fhj/sva-var/rtl_analyzer/analyzer_cpp

# 2. 清理
rm -rf build

# 3. 配置
cmake -B build

# 4. 编译
cmake --build build -j8

# 5. 运行
./build/analyzer