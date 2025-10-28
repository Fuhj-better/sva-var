import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Set, Any, Tuple
import statistics
import time
import re

class EnhancedVariableScoringSystem:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.data = None
        self.graph = nx.DiGraph()
        self.control_graph = nx.DiGraph()
        self.variable_scores = {}
        self.control_signals_analysis = {}
        
        # 新增分析结果存储
        self.timing_paths_analysis = {}
        self.fanout_analysis = {}
        self.global_signals_analysis = {}
        self.fsm_analysis = {}
        self.module_interface_analysis = {}
        self.critical_paths = []
        self.high_fanout_signals = []
        self.fsm_candidates = []
        
        # 添加缓存
        self.betweenness_centrality_cached = {}
        
        self.load_data()
        
    def load_data(self):
        """加载JSON数据"""
        with open(self.json_file_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} variables")
    
    def build_dependency_graph(self):
        """构建详细的变量依赖图，包括数据依赖和控制依赖"""
        print("Building dependency graphs...")
        
        # 构建数据依赖图
        for var_name, var_info in self.data.items():
            self.graph.add_node(var_name, **var_info)
            
            # 添加赋值依赖边
            for assignment in var_info.get('assignments', []):
                for driving_signal in assignment.get('drivingSignals', []):
                    if driving_signal in self.data:
                        edge_attrs = {
                            'type': 'assignment',
                            'logic_type': assignment.get('logicType', 'unknown'),
                            'condition_depth': assignment.get('conditionDepth', 0),
                            'assignment_type': assignment.get('type', 'direct')
                        }
                        self.graph.add_edge(driving_signal, var_name, **edge_attrs)
            
            # 添加fanout依赖边
            for fanout_var in var_info.get('fanOut', []):
                if fanout_var in self.data:
                    self.graph.add_edge(var_name, fanout_var, type='fanout')
        
        print(f"Data dependency graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # 构建控制依赖图
        self.build_control_dependency_graph()
    
    def build_control_dependency_graph(self):
        """构建控制依赖图"""
        print("Building control dependency graph...")
        
        for var_name, var_info in self.data.items():
            self.control_graph.add_node(var_name)
            
            control_signals_for_var = set()
            
            for assignment in var_info.get('assignments', []):
                control_signals = self.extract_control_signals_from_path(assignment.get('path', []))
                control_signals_for_var.update(control_signals)
                
                for control_signal in control_signals:
                    if control_signal in self.data:
                        if not self.control_graph.has_edge(control_signal, var_name):
                            self.control_graph.add_edge(control_signal, var_name, weight=1)
                        else:
                            self.control_graph[control_signal][var_name]['weight'] += 1
            
            self.control_signals_analysis[var_name] = {
                'controlled_by': list(control_signals_for_var),
                'control_frequency': len(control_signals_for_var)
            }
        
        print(f"Control dependency graph built: {self.control_graph.number_of_nodes()} nodes, {self.control_graph.number_of_edges()} edges")
    
    def extract_control_signals_from_path(self, path):
        """从条件路径中提取控制信号"""
        control_signals = set()
        
        for clause in path:
            expr_info = clause.get('expr', {})
            signals = expr_info.get('involvedSignals', [])
            control_signals.update(signals)
        
        return control_signals

    # =====================================================================
    # 新增功能 1: 时序路径分析 ⭐⭐⭐⭐⭐
    # =====================================================================
    
    def analyze_timing_paths(self):
        """分析关键时序路径 - 对性能至关重要"""
        print("\n" + "="*80)
        print("TIMING PATH ANALYSIS ⭐⭐⭐⭐⭐")
        print("="*80)
        
        start_time = time.time()
        
        # 识别所有时序元素（寄存器）
        sequential_vars = [
            var_name for var_name, var_info in self.data.items()
            if any(a.get('logicType') == 'sequential' for a in var_info.get('assignments', []))
        ]
        
        print(f"Found {len(sequential_vars)} sequential elements")
        
        # 分析每个寄存器的路径深度
        for var_name in sequential_vars:
            analysis = self.analyze_register_paths(var_name)
            self.timing_paths_analysis[var_name] = analysis
        
        # 识别关键路径（最长组合逻辑路径）
        self.critical_paths = self.identify_critical_paths()
        
        end_time = time.time()
        print(f"Timing path analysis completed in {end_time - start_time:.2f} seconds")
        
        return self.critical_paths
    
    def analyze_register_paths(self, reg_name):
        """分析从寄存器出发的路径"""
        analysis = {
            'max_comb_depth_forward': 0,
            'max_comb_depth_backward': 0,
            'fanout_stages': 0,
            'drives_outputs': False,
            'critical_path_score': 0.0,
            'is_critical_register': False
        }
        
        # 前向路径分析
        forward_depth = self.calculate_combinational_depth(reg_name, direction='forward')
        analysis['max_comb_depth_forward'] = forward_depth
        
        # 后向路径分析
        backward_depth = self.calculate_combinational_depth(reg_name, direction='backward')
        analysis['max_comb_depth_backward'] = backward_depth
        
        # 检查是否直接驱动输出
        var_info = self.data.get(reg_name, {})
        analysis['drives_outputs'] = var_info.get('drivesOutput', False)
        
        # 计算关键路径评分
        analysis['critical_path_score'] = (forward_depth * 0.6 + backward_depth * 0.4)
        analysis['is_critical_register'] = analysis['critical_path_score'] > 5.0
        
        return analysis
    
    def calculate_combinational_depth(self, start_var, direction='forward', max_depth=20):
        """计算组合逻辑深度（不经过时序元素）"""
        visited = set()
        max_comb_depth = 0
        
        def dfs(var, depth):
            nonlocal max_comb_depth
            
            if depth > max_depth or var in visited:
                return
            
            visited.add(var)
            var_info = self.data.get(var, {})
            
            # 检查是否是时序逻辑
            is_sequential = any(
                a.get('logicType') == 'sequential' 
                for a in var_info.get('assignments', [])
            )
            
            # 如果遇到时序逻辑且不是起点，停止
            if is_sequential and var != start_var:
                max_comb_depth = max(max_comb_depth, depth)
                return
            
            # 继续遍历
            if direction == 'forward':
                neighbors = list(self.graph.successors(var))
            else:
                neighbors = list(self.graph.predecessors(var))
            
            if not neighbors:
                max_comb_depth = max(max_comb_depth, depth)
            else:
                for neighbor in neighbors:
                    dfs(neighbor, depth + 1)
        
        dfs(start_var, 0)
        return max_comb_depth
    
    def identify_critical_paths(self, top_n=15):
        """识别Top-N关键路径"""
        critical_paths = []
        
        for var_name, analysis in self.timing_paths_analysis.items():
            score = analysis['critical_path_score']
            if score > 0:
                critical_paths.append((var_name, score, analysis))
        
        critical_paths.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔴 Top {top_n} Critical Timing Paths (Performance Bottlenecks):")
        print("-" * 80)
        for i, (var_name, score, analysis) in enumerate(critical_paths[:top_n]):
            critical_marker = " ⚠️" if analysis['is_critical_register'] else ""
            print(f"{i+1:2d}. {var_name}{critical_marker}")
            print(f"    Critical Score: {score:.2f}")
            print(f"    Forward Depth: {analysis['max_comb_depth_forward']}")
            print(f"    Backward Depth: {analysis['max_comb_depth_backward']}")
            print(f"    Drives Output: {analysis['drives_outputs']}")
        
        return critical_paths[:top_n]

    # =====================================================================
    # 新增功能 2: 扇出负载分析 ⭐⭐⭐⭐
    # =====================================================================
    
    def analyze_fanout_load(self):
        """分析扇出负载 - 高扇出信号是设计瓶颈"""
        print("\n" + "="*80)
        print("FANOUT LOAD ANALYSIS ⭐⭐⭐⭐")
        print("="*80)
        
        start_time = time.time()
        
        for var_name, var_info in self.data.items():
            fanout_list = var_info.get('fanOut', [])
            fanout_count = len(fanout_list)
            
            # 分析扇出的层次结构
            fanout_depth = self.calculate_fanout_depth(var_name)
            
            # 分析扇出的类型分布
            fanout_types = self.analyze_fanout_types(var_name, fanout_list)
            
            # 计算有效扇出（去除中间节点）
            effective_fanout = self.calculate_effective_fanout(var_name)
            
            self.fanout_analysis[var_name] = {
                'direct_fanout': fanout_count,
                'fanout_depth': fanout_depth,
                'effective_fanout': effective_fanout,
                'fanout_types': fanout_types,
                'fanout_pressure': self.calculate_fanout_pressure(fanout_count, fanout_depth),
                'is_high_fanout': fanout_count > 10,
                'is_critical_fanout': fanout_count > 20  # 更高阈值
            }
        
        # 识别高扇出瓶颈
        self.high_fanout_signals = self.identify_high_fanout_bottlenecks()
        
        end_time = time.time()
        print(f"Fanout analysis completed in {end_time - start_time:.2f} seconds")
        
        return self.high_fanout_signals
    
    def calculate_fanout_depth(self, var_name, max_depth=5):
        """计算扇出的最大深度"""
        visited = set()
        max_depth_found = 0
        
        def dfs(var, depth):
            nonlocal max_depth_found
            
            if depth > max_depth or var in visited:
                return
            
            visited.add(var)
            max_depth_found = max(max_depth_found, depth)
            
            var_info = self.data.get(var, {})
            fanout_list = var_info.get('fanOut', [])
            
            for fanout_var in fanout_list:
                if fanout_var in self.data:
                    dfs(fanout_var, depth + 1)
        
        dfs(var_name, 0)
        return max_depth_found
    
    def analyze_fanout_types(self, var_name, fanout_list):
        """分析扇出目标的类型分布"""
        types = {
            'sequential': 0,
            'combinational': 0,
            'output': 0,
            'control': 0
        }
        
        for fanout_var in fanout_list:
            if fanout_var not in self.data:
                continue
            
            var_info = self.data[fanout_var]
            
            # 检查是否是输出
            if var_info.get('direction') == 'output':
                types['output'] += 1
            
            # 检查是否是控制变量
            if var_info.get('isControlVariable', False):
                types['control'] += 1
            
            # 检查逻辑类型
            assignments = var_info.get('assignments', [])
            if any(a.get('logicType') == 'sequential' for a in assignments):
                types['sequential'] += 1
            else:
                types['combinational'] += 1
        
        return types
    
    def calculate_effective_fanout(self, var_name):
        """计算有效扇出（到达输出或寄存器的路径数）"""
        effective_count = 0
        visited = set()
        
        def dfs(var):
            nonlocal effective_count
            
            if var in visited:
                return
            visited.add(var)
            
            var_info = self.data.get(var, {})
            if not var_info:
                return
            
            # 如果是输出或寄存器，计数
            is_output = var_info.get('direction') == 'output'
            is_sequential = any(
                a.get('logicType') == 'sequential' 
                for a in var_info.get('assignments', [])
            )
            
            if is_output or (is_sequential and var != var_name):
                effective_count += 1
                return
            
            # 继续遍历
            for fanout_var in var_info.get('fanOut', []):
                if fanout_var in self.data:
                    dfs(fanout_var)
        
        dfs(var_name)
        return effective_count
    
    def calculate_fanout_pressure(self, fanout_count, fanout_depth):
        """计算扇出压力指标"""
        return fanout_count * (1 + fanout_depth * 0.5)
    
    def identify_high_fanout_bottlenecks(self, top_n=15):
        """识别高扇出瓶颈"""
        bottlenecks = []
        
        for var_name, analysis in self.fanout_analysis.items():
            if analysis['is_high_fanout']:
                pressure = analysis['fanout_pressure']
                bottlenecks.append((var_name, pressure, analysis))
        
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🔴 Top {top_n} High Fanout Bottlenecks (Timing Risks):")
        print("-" * 80)
        for i, (var_name, pressure, analysis) in enumerate(bottlenecks[:top_n]):
            critical_marker = " ⚡" if analysis['is_critical_fanout'] else ""
            print(f"{i+1:2d}. {var_name}{critical_marker}")
            print(f"    Fanout Pressure: {pressure:.2f}")
            print(f"    Direct Fanout: {analysis['direct_fanout']}")
            print(f"    Effective Fanout: {analysis['effective_fanout']}")
            print(f"    Fanout Depth: {analysis['fanout_depth']}")
            print(f"    Types: Seq={analysis['fanout_types']['sequential']}, "
                  f"Comb={analysis['fanout_types']['combinational']}, "
                  f"Out={analysis['fanout_types']['output']}")
        
        return bottlenecks[:top_n]

    # =====================================================================
    # 新增功能 3: 复位/时钟全局信号分析 ⭐⭐⭐⭐
    # =====================================================================
    
    def analyze_global_signals(self):
        """分析复位、时钟等全局信号 - 需要特殊关注"""
        print("\n" + "="*80)
        print("GLOBAL SIGNALS ANALYSIS (Clock/Reset) ⭐⭐⭐⭐")
        print("="*80)
        
        start_time = time.time()
        
        # 识别模式
        clock_patterns = [r'clk', r'clock', r'ck']
        reset_patterns = [r'rst', r'reset', r'rstn', r'rst_ni?']
        enable_patterns = [r'_en$', r'enable', r'_ce$', r'valid', r'ready']
        
        for var_name, var_info in self.data.items():
            var_lower = var_name.lower()
            
            signal_type = 'regular'
            
            # 检查是否是时钟
            if any(re.search(pattern, var_lower) for pattern in clock_patterns):
                signal_type = 'clock'
            # 检查是否是复位
            elif any(re.search(pattern, var_lower) for pattern in reset_patterns):
                signal_type = 'reset'
            # 检查是否是使能或握手信号
            elif any(re.search(pattern, var_lower) for pattern in enable_patterns):
                signal_type = 'enable'
            
            # 分析全局信号的影响范围
            if signal_type != 'regular':
                analysis = self.analyze_global_signal_impact(var_name, signal_type)
                self.global_signals_analysis[var_name] = analysis
        
        # 生成报告
        self.report_global_signals()
        
        end_time = time.time()
        print(f"Global signals analysis completed in {end_time - start_time:.2f} seconds")
    
    def analyze_global_signal_impact(self, signal_name, signal_type):
        """分析全局信号的影响"""
        var_info = self.data.get(signal_name, {})
        
        # 直接影响
        direct_fanout = len(var_info.get('fanOut', []))
        
        # 影响的寄存器数量
        affected_registers = 0
        affected_outputs = 0
        control_scope = 0
        
        for fanout_var in var_info.get('fanOut', []):
            if fanout_var not in self.data:
                continue
            
            fanout_info = self.data[fanout_var]
            
            # 统计影响的寄存器
            if any(a.get('logicType') == 'sequential' for a in fanout_info.get('assignments', [])):
                affected_registers += 1
            
            # 统计影响的输出
            if fanout_info.get('direction') == 'output':
                affected_outputs += 1
        
        # 控制影响范围（通过控制图）
        if signal_name in self.control_graph:
            control_scope = self.control_graph.out_degree(signal_name)
        
        return {
            'signal_type': signal_type,
            'direct_fanout': direct_fanout,
            'affected_registers': affected_registers,
            'affected_outputs': affected_outputs,
            'control_scope': control_scope,
            'criticality': self.calculate_global_signal_criticality(
                signal_type, direct_fanout, affected_registers
            )
        }
    
    def calculate_global_signal_criticality(self, signal_type, fanout, affected_regs):
        """计算全局信号的关键性"""
        base_score = {
            'clock': 10.0,
            'reset': 9.0,
            'enable': 7.0
        }.get(signal_type, 5.0)
        
        # 根据影响范围调整
        fanout_factor = min(fanout / 100.0, 2.0)
        reg_factor = min(affected_regs / 50.0, 1.5)
        
        return base_score * (1 + fanout_factor + reg_factor)
    
    def report_global_signals(self):
        """生成全局信号报告"""
        if not self.global_signals_analysis:
            print("No global signals identified.")
            return
        
        # 按类型分组
        by_type = defaultdict(list)
        for sig_name, analysis in self.global_signals_analysis.items():
            by_type[analysis['signal_type']].append((sig_name, analysis))
        
        for sig_type in ['clock', 'reset', 'enable']:
            signals = by_type.get(sig_type, [])
            if not signals:
                continue
            
            signals.sort(key=lambda x: x[1]['criticality'], reverse=True)
            
            print(f"\n🔵 {sig_type.upper()} Signals ({len(signals)} found):")
            print("-" * 80)
            
            for sig_name, analysis in signals[:8]:  # 显示前8个
                critical_marker = " 🚨" if analysis['criticality'] > 15 else ""
                print(f"  • {sig_name}{critical_marker}")
                print(f"    Criticality: {analysis['criticality']:.2f}")
                print(f"    Direct Fanout: {analysis['direct_fanout']}")
                print(f"    Affected Registers: {analysis['affected_registers']}")
                print(f"    Control Scope: {analysis['control_scope']}")

    # =====================================================================
    # 新增功能 4: 状态机识别 ⭐⭐⭐
    # =====================================================================
    
    def identify_state_machines(self):
        """识别状态机 - FSM变量通常是控制核心"""
        print("\n" + "="*80)
        print("STATE MACHINE IDENTIFICATION ⭐⭐⭐")
        print("="*80)
        
        start_time = time.time()
        
        # 状态机识别启发式规则
        self.fsm_candidates = []
        
        for var_name, var_info in self.data.items():
            score = self.calculate_fsm_likelihood(var_name, var_info)
            
            if score > 0.5:  # 阈值
                fsm_analysis = self.analyze_fsm(var_name, var_info)
                self.fsm_analysis[var_name] = fsm_analysis
                self.fsm_candidates.append((var_name, score, fsm_analysis))
        
        self.fsm_candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🟢 Identified {len(self.fsm_candidates)} potential state machines:")
        print("-" * 80)
        
        for i, (var_name, score, analysis) in enumerate(self.fsm_candidates[:10]):
            fsm_type = "Mealy" if analysis['is_mealy'] else "Moore"
            print(f"{i+1:2d}. {var_name}")
            print(f"    FSM Likelihood: {score:.2f}")
            print(f"    Type: {fsm_type}")
            print(f"    Estimated States: {analysis['estimated_states']}")
            print(f"    Control Outputs: {analysis['control_outputs']}")
            print(f"    Transition Complexity: {analysis['transition_complexity']}")
        
        end_time = time.time()
        print(f"FSM identification completed in {end_time - start_time:.2f} seconds")
        
        return self.fsm_candidates
    
    def calculate_fsm_likelihood(self, var_name, var_info):
        """计算变量是状态机的可能性"""
        score = 0.0
        
        # 规则1: 名字包含state/fsm
        if any(pattern in var_name.lower() for pattern in ['state', 'fsm', 'st_']):
            score += 0.4
        
        # 规则2: 是时序逻辑
        is_sequential = any(
            a.get('logicType') == 'sequential' 
            for a in var_info.get('assignments', [])
        )
        if is_sequential:
            score += 0.2
        
        # 规则3: 有自反馈（状态转移）
        has_self_feedback = False
        for assignment in var_info.get('assignments', []):
            driving_signals = assignment.get('drivingSignals', [])
            if var_name in driving_signals:
                has_self_feedback = True
                score += 0.3
                break
        
        # 规则4: 控制很多其他信号
        if var_name in self.control_graph:
            control_out = self.control_graph.out_degree(var_name)
            if control_out > 5:
                score += 0.2
        
        # 规则5: 多路选择赋值
        assignments = var_info.get('assignments', [])
        if len(assignments) > 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def analyze_fsm(self, var_name, var_info):
        """分析状态机特征"""
        analysis = {
            'estimated_states': 0,
            'control_outputs': 0,
            'has_self_feedback': False,
            'transition_complexity': 0,
            'is_mealy': False,
            'is_moore': False
        }
        
        # 估计状态数（通过赋值路径数）
        assignments = var_info.get('assignments', [])
        analysis['estimated_states'] = len(assignments)
        
        # 控制输出数量
        if var_name in self.control_graph:
            analysis['control_outputs'] = self.control_graph.out_degree(var_name)
        
        # 检查自反馈
        for assignment in assignments:
            if var_name in assignment.get('drivingSignals', []):
                analysis['has_self_feedback'] = True
                break
        
        # 转移复杂度
        total_conditions = sum(
            len(a.get('path', [])) for a in assignments
        )
        analysis['transition_complexity'] = total_conditions
        
        # 判断FSM类型
        input_dependency_count = 0
        for assignment in assignments:
            for ds in assignment.get('drivingSignals', []):
                ds_info = self.data.get(ds, {})
                if ds_info.get('direction') == 'input':
                    input_dependency_count += 1
        
        if input_dependency_count > len(assignments) / 2:
            analysis['is_mealy'] = True
        else:
            analysis['is_moore'] = True
        
        return analysis

    # =====================================================================
    # 新增功能 5: 跨模块依赖分析 ⭐⭐⭐
    # =====================================================================
    
    def analyze_module_interfaces(self):
        """分析跨模块依赖 - 模块间接口信号重要性高"""
        print("\n" + "="*80)
        print("MODULE INTERFACE ANALYSIS ⭐⭐⭐")
        print("="*80)
        
        start_time = time.time()
        
        # 识别所有接口信号
        interface_signals = {
            'inputs': [],
            'outputs': [],
            'inouts': []
        }
        
        for var_name, var_info in self.data.items():
            direction = var_info.get('direction', '')
            if direction in ['input', 'output', 'inout']:
                interface_signals[direction + 's'].append(var_name)
                
                # 分析每个接口信号
                analysis = self.analyze_interface_signal(var_name, var_info, direction)
                self.module_interface_analysis[var_name] = analysis
        
        # 生成接口报告
        self.report_module_interfaces(interface_signals)
        
        end_time = time.time()
        print(f"Module interface analysis completed in {end_time - start_time:.2f} seconds")
    
    def analyze_interface_signal(self, sig_name, sig_info, direction):
        """分析单个接口信号"""
        analysis = {
            'direction': direction,
            'bit_width': sig_info.get('bitWidth', 1),
            'internal_fanout': 0,
            'internal_fanin': 0,
            'is_control_interface': False,
            'is_data_interface': False,
            'complexity_score': 0.0,
            'is_critical_interface': False
        }
        
        # 内部扇出/扇入
        if direction == 'input':
            analysis['internal_fanout'] = len(sig_info.get('fanOut', []))
        elif direction == 'output':
            analysis['internal_fanin'] = len(sig_info.get('assignments', []))
        
        # 判断是控制接口还是数据接口
        if sig_name in self.control_graph:
            control_degree = self.control_graph.out_degree(sig_name) if direction == 'input' else self.control_graph.in_degree(sig_name)
            if control_degree > 3:
                analysis['is_control_interface'] = True
        
        if analysis['bit_width'] > 1 and not analysis['is_control_interface']:
            analysis['is_data_interface'] = True
        
        # 复杂度评分
        if direction == 'input':
            analysis['complexity_score'] = analysis['internal_fanout'] * (1 + analysis['bit_width'] / 10.0)
        else:
            analysis['complexity_score'] = analysis['internal_fanin'] * (1 + analysis['bit_width'] / 10.0)
        
        analysis['is_critical_interface'] = analysis['complexity_score'] > 15.0
        
        return analysis
    
    def report_module_interfaces(self, interface_signals):
        """生成模块接口报告"""
        print(f"\n🟣 Interface Summary:")
        print(f"  Inputs: {len(interface_signals['inputs'])}")
        print(f"  Outputs: {len(interface_signals['outputs'])}")
        print(f"  Inouts: {len(interface_signals['inouts'])}")
        
        # 识别关键接口
        critical_interfaces = []
        for sig_name, analysis in self.module_interface_analysis.items():
            if analysis['is_critical_interface']:
                critical_interfaces.append((sig_name, analysis))
        
        critical_interfaces.sort(key=lambda x: x[1]['complexity_score'], reverse=True)
        
        print(f"\n🔴 Top Critical Interface Signals:")
        print("-" * 80)
        for i, (sig_name, analysis) in enumerate(critical_interfaces[:10]):
            interface_type = "Control" if analysis['is_control_interface'] else "Data"
            critical_marker = " 🔥" if analysis['complexity_score'] > 30 else ""
            print(f"{i+1:2d}. {sig_name}{critical_marker}")
            print(f"    Type: {interface_type} {analysis['direction']}")
            print(f"    Complexity Score: {analysis['complexity_score']:.2f}")
            print(f"    Bit Width: {analysis['bit_width']}")
            if analysis['direction'] == 'input':
                print(f"    Internal Fanout: {analysis['internal_fanout']}")
            else:
                print(f"    Internal Fanin: {analysis['internal_fanin']}")

    # =====================================================================
    # 综合评分和报告生成
    # =====================================================================
    
    def run_comprehensive_analysis(self):
        """运行完整的综合分析"""
        print("🚀 Starting Comprehensive Variable Analysis")
        print("=" * 80)
        
        # 构建基础依赖图
        self.build_dependency_graph()
        
        # 运行所有分析并验证
        analyses = [
            ('Timing Paths', self.analyze_timing_paths),
            ('Fanout Load', self.analyze_fanout_load),
            ('Global Signals', self.analyze_global_signals),
            ('State Machines', self.identify_state_machines),
            ('Module Interfaces', self.analyze_module_interfaces)
        ]
        
        for name, analysis_func in analyses:
            print(f"\n🔧 Running {name} Analysis...")
            try:
                result = analysis_func()
                print(f"✅ {name} Analysis Completed")
            except Exception as e:
                print(f"❌ {name} Analysis Failed: {e}")
        
        # 计算基础指标
        print(f"\n🔧 Calculating Base Metrics...")
        self.calculate_all_metrics_fast()
        
        # 归一化评分
        print(f"\n🔧 Normalizing Scores...")
        self.normalize_scores()
        
        # 整合增强分析结果
        print(f"\n🔧 Integrating Enhanced Metrics...")
        self.integrate_enhanced_metrics()
        
        # 生成综合报告
        self.generate_comprehensive_report()
        
        # 导出完整结果
        self.export_comprehensive_results('comprehensive_analysis_results.json')
        
        print("\n✅ All analyses completed!")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
        print("="*80)
        
        # 关键发现统计
        print(f"\n📈 Key Findings:")
        print(f"  • Critical Timing Paths: {len(self.critical_paths)}")
        print(f"  • High Fanout Signals: {len(self.high_fanout_signals)}")
        print(f"  • Global Signals: {len(self.global_signals_analysis)}")
        print(f"  • State Machines: {len(self.fsm_candidates)}")
        print(f"  • Critical Interfaces: {len([v for v in self.module_interface_analysis.values() if v['is_critical_interface']])}")
        
        # 设计质量指标
        print(f"\n🎯 Design Quality Indicators:")
        
        # 平均组合逻辑深度
        avg_comb_depth = np.mean([a['critical_path_score'] for a in self.timing_paths_analysis.values()])
        print(f"  • Average Combinational Depth: {avg_comb_depth:.2f}")
        
        # 高扇出信号比例
        high_fanout_ratio = len(self.high_fanout_signals) / len(self.data) * 100
        print(f"  • High Fanout Signals: {high_fanout_ratio:.1f}%")
        
        # 控制复杂度
        avg_control_out = np.mean([self.control_graph.out_degree(var) for var in self.data.keys() if var in self.control_graph])
        print(f"  • Average Control Out Degree: {avg_control_out:.2f}")
        
        print(f"\n💡 Verification Priority Recommendations:")
        print(f"  1. Focus on {len(self.critical_paths)} critical timing paths")
        print(f"  2. Address {len(self.high_fanout_signals)} high fanout bottlenecks") 
        print(f"  3. Verify {len(self.fsm_candidates)} state machine behaviors")
        print(f"  4. Test global signal integrity")
        print(f"  5. Validate critical module interfaces")

    def integrate_enhanced_metrics(self):
        """将增强分析结果整合到variable_scores中"""
        print("\nIntegrating enhanced analysis results...")
        
        for var_name in self.data.keys():
            # 确保variable_scores中存在该变量
            if var_name not in self.variable_scores:
                self.variable_scores[var_name] = {}
            
            scores = self.variable_scores[var_name]
            
            # 整合时序路径分析
            if var_name in self.timing_paths_analysis:
                timing_analysis = self.timing_paths_analysis[var_name]
                scores.update({
                    'critical_path_score': timing_analysis['critical_path_score'],
                    'max_comb_depth_forward': timing_analysis['max_comb_depth_forward'],
                    'max_comb_depth_backward': timing_analysis['max_comb_depth_backward'],
                    'is_critical_register': timing_analysis['is_critical_register'],
                    'drives_outputs': timing_analysis['drives_outputs']
                })
            else:
                # 对于非寄存器变量，设置默认值
                scores.update({
                    'critical_path_score': 0.0,
                    'max_comb_depth_forward': 0,
                    'max_comb_depth_backward': 0,
                    'is_critical_register': False,
                    'drives_outputs': False
                })
            
            # 整合扇出负载分析
            if var_name in self.fanout_analysis:
                fanout_analysis = self.fanout_analysis[var_name]
                scores.update({
                    'fanout_pressure': fanout_analysis['fanout_pressure'],
                    'direct_fanout': fanout_analysis['direct_fanout'],
                    'effective_fanout': fanout_analysis['effective_fanout'],
                    'fanout_depth': fanout_analysis['fanout_depth'],
                    'is_high_fanout': fanout_analysis['is_high_fanout'],
                    'is_critical_fanout': fanout_analysis['is_critical_fanout']
                })
            else:
                # 设置默认值
                var_info = self.data.get(var_name, {})
                fanout_count = len(var_info.get('fanOut', []))
                scores.update({
                    'fanout_pressure': 0.0,
                    'direct_fanout': fanout_count,
                    'effective_fanout': 0,
                    'fanout_depth': 0,
                    'is_high_fanout': False,
                    'is_critical_fanout': False
                })
            
            # 整合全局信号分析
            if var_name in self.global_signals_analysis:
                global_analysis = self.global_signals_analysis[var_name]
                scores.update({
                    'global_signal_criticality': global_analysis['criticality'],
                    'signal_type': global_analysis['signal_type'],
                    'affected_registers': global_analysis['affected_registers'],
                    'global_direct_fanout': global_analysis['direct_fanout'],
                    'global_control_scope': global_analysis['control_scope']
                })
            else:
                # 设置默认值
                scores.update({
                    'global_signal_criticality': 0.0,
                    'signal_type': 'regular',
                    'affected_registers': 0,
                    'global_direct_fanout': 0,
                    'global_control_scope': 0
                })
            
            # 整合状态机分析
            if var_name in self.fsm_analysis:
                fsm_analysis = self.fsm_analysis[var_name]
                fsm_score = self.get_fsm_score(var_name)
                scores.update({
                    'fsm_likelihood': fsm_score,
                    'estimated_states': fsm_analysis['estimated_states'],
                    'transition_complexity': fsm_analysis['transition_complexity'],
                    'fsm_type': 'mealy' if fsm_analysis['is_mealy'] else 'moore',
                    'fsm_control_outputs': fsm_analysis['control_outputs'],
                    'has_self_feedback': fsm_analysis['has_self_feedback']
                })
            else:
                # 设置默认值
                scores.update({
                    'fsm_likelihood': 0.0,
                    'estimated_states': 0,
                    'transition_complexity': 0,
                    'fsm_type': 'none',
                    'fsm_control_outputs': 0,
                    'has_self_feedback': False
                })
            
            # 整合接口分析
            if var_name in self.module_interface_analysis:
                interface_analysis = self.module_interface_analysis[var_name]
                scores.update({
                    'interface_complexity': interface_analysis['complexity_score'],
                    'is_critical_interface': interface_analysis['is_critical_interface'],
                    'interface_type': 'control' if interface_analysis['is_control_interface'] else 'data',
                    'internal_fanout': interface_analysis['internal_fanout'],
                    'internal_fanin': interface_analysis['internal_fanin']
                })
            else:
                # 设置默认值
                var_info = self.data.get(var_name, {})
                direction = var_info.get('direction', '')
                if direction == 'input':
                    internal_fanout = len(var_info.get('fanOut', []))
                    scores.update({
                        'interface_complexity': 0.0,
                        'is_critical_interface': False,
                        'interface_type': 'none',
                        'internal_fanout': internal_fanout,
                        'internal_fanin': 0
                    })
                elif direction == 'output':
                    internal_fanin = len(var_info.get('assignments', []))
                    scores.update({
                        'interface_complexity': 0.0,
                        'is_critical_interface': False,
                        'interface_type': 'none',
                        'internal_fanout': 0,
                        'internal_fanin': internal_fanin
                    })
                else:
                    scores.update({
                        'interface_complexity': 0.0,
                        'is_critical_interface': False,
                        'interface_type': 'none',
                        'internal_fanout': 0,
                        'internal_fanin': 0
                    })

    def get_fsm_score(self, var_name):
        """获取状态机评分"""
        for candidate in self.fsm_candidates:
            if candidate[0] == var_name:
                return candidate[1]  # 返回FSM可能性评分
        return 0.0

    def export_comprehensive_results(self, output_file):
        """导出包含所有分析结果的完整JSON"""
        # 整合所有结果
        comprehensive_results = {
            'metadata': {
                'total_variables': len(self.variable_scores),
                'data_graph_nodes': self.graph.number_of_nodes(),
                'data_graph_edges': self.graph.number_of_edges(),
                'control_graph_nodes': self.control_graph.number_of_nodes(),
                'control_graph_edges': self.control_graph.number_of_edges(),
                'critical_timing_paths': len(self.critical_paths),
                'high_fanout_signals': len(self.high_fanout_signals),
                'state_machines': len(self.fsm_candidates),
                'global_signals': len(self.global_signals_analysis)
            },
            'variable_scores': self.variable_scores,
            'analysis_summary': {
                'critical_paths': self.critical_paths,
                'high_fanout_signals': self.high_fanout_signals,
                'fsm_candidates': self.fsm_candidates,
                'global_signals_analysis': self.global_signals_analysis
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_results, f, indent=2, ensure_ascii=False)
        
        print(f"Comprehensive results exported to: {output_file}")

    def calculate_all_metrics_fast(self):
        """快速计算所有基础指标"""
        print("Calculating base metrics...")
        
        total_nodes = len(self.data)
        processed = 0
        
        for var_name in self.data.keys():
            scores = {}
            
            # 计算复杂度维度
            scores.update(self.calculate_complexity_metrics(var_name))
            
            # 计算中心性维度
            scores.update(self.calculate_centrality_metrics(var_name))
            
            # 计算结构维度
            scores.update(self.calculate_structural_metrics(var_name))
            
            # 计算功能维度
            scores.update(self.calculate_functional_metrics(var_name))
            
            # 计算可靠性维度
            scores.update(self.calculate_reliability_metrics(var_name))
            
            self.variable_scores[var_name] = scores
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_nodes} variables...")
        
        print(f"Base metrics calculation completed for {len(self.variable_scores)} variables")

    def calculate_complexity_metrics(self, var_name):
        """计算复杂度相关指标"""
        var_info = self.data[var_name]
        assignments = var_info.get('assignments', [])
        
        metrics = {}
        
        # 1. 条件复杂度
        condition_depths = [assign.get('conditionDepth', 0) for assign in assignments]
        metrics['max_condition_depth'] = max(condition_depths) if condition_depths else 0
        metrics['avg_condition_depth'] = statistics.mean(condition_depths) if condition_depths else 0
        
        # 2. 路径多样性
        unique_paths = len(set(str(assign.get('path', [])) for assign in assignments))
        metrics['path_diversity'] = unique_paths
        
        # 3. 逻辑类型复杂度
        logic_types = [assign.get('logicType', 'unknown') for assign in assignments]
        logic_type_counts = Counter(logic_types)
        metrics['sequential_assignments'] = logic_type_counts.get('sequential', 0)
        metrics['combinational_assignments'] = logic_type_counts.get('combinational', 0)
        metrics['logic_type_mix'] = len(set(logic_types))
        
        # 4. 赋值类型复杂度
        assignment_types = [assign.get('type', 'direct') for assign in assignments]
        metrics['assignment_type_diversity'] = len(set(assignment_types))
        
        return metrics

    def calculate_centrality_metrics(self, var_name):
        """计算中心性相关指标 - 修复版本"""
        metrics = {}
        
        # 1. 度数中心性
        try:
            in_degree = self.graph.in_degree(var_name)
            out_degree = self.graph.out_degree(var_name)
            metrics['in_degree'] = in_degree
            metrics['out_degree'] = out_degree
            metrics['total_degree'] = in_degree + out_degree
        except:
            metrics['in_degree'] = 0
            metrics['out_degree'] = 0
            metrics['total_degree'] = 0
        
        # 2. 介数中心性 (改进计算)
        try:
            # 对于大图，使用基于度数的近似
            total_nodes = self.graph.number_of_nodes()
            if total_nodes > 1000:  # 大图使用近似
                metrics['betweenness_centrality'] = metrics['total_degree'] / (total_nodes - 1)
            else:
                # 小图可以计算精确值
                if not hasattr(self, 'betweenness_centrality_cached'):
                    print("Calculating betweenness centrality...")
                    self.betweenness_centrality_cached = nx.betweenness_centrality(self.graph, k=min(100, total_nodes))
                metrics['betweenness_centrality'] = self.betweenness_centrality_cached.get(var_name, 0.0)
        except:
            metrics['betweenness_centrality'] = 0.0
        
        # 3. 接近中心性 (改进计算)
        try:
            # 使用连通分量内的计算
            if self.graph.number_of_nodes() > 500:
                # 大图使用简化计算
                neighbors = set(self.graph.predecessors(var_name)) | set(self.graph.successors(var_name))
                metrics['closeness_centrality'] = len(neighbors) / self.graph.number_of_nodes()
            else:
                # 小图计算精确值
                connected_component = self.graph.subgraph(nx.node_connected_component(self.graph.to_undirected(), var_name))
                if len(connected_component) > 1:
                    closeness = nx.closeness_centrality(connected_component, var_name)
                    metrics['closeness_centrality'] = closeness
                else:
                    metrics['closeness_centrality'] = 0.0
        except:
            metrics['closeness_centrality'] = 0.0
        
        return metrics

    def calculate_structural_metrics(self, var_name):
        """计算结构重要性指标 - 修复版本"""
        metrics = {}
        
        # 1. 控制影响范围
        try:
            if var_name in self.control_graph:
                control_descendants = len(nx.descendants(self.control_graph, var_name))
                metrics['control_scope'] = control_descendants
            else:
                metrics['control_scope'] = 0
        except:
            metrics['control_scope'] = 0
        
        # 2. 最大影响深度 (改进计算)
        try:
            # 使用图遍历计算实际深度
            max_depth = 0
            visited = set()
            
            def dfs(current, depth):
                nonlocal max_depth
                if depth > 10 or current in visited:  # 限制深度避免无限循环
                    return
                visited.add(current)
                max_depth = max(max_depth, depth)
                
                # 遍历后继节点
                for successor in self.graph.successors(current):
                    if successor != var_name:  # 避免自环
                        dfs(successor, depth + 1)
            
            dfs(var_name, 0)
            metrics['max_influence_depth'] = max_depth
        except:
            metrics['max_influence_depth'] = 0
        
        # 3. 依赖广度 (修复计算)
        try:
            var_info = self.data.get(var_name, {})
            all_driving_signals = set()
            
            # 统计所有赋值中的驱动信号
            for assignment in var_info.get('assignments', []):
                driving_signals = assignment.get('drivingSignals', [])
                all_driving_signals.update(driving_signals)
                
                # 同时统计条件路径中的信号
                for clause in assignment.get('path', []):
                    expr_info = clause.get('expr', {})
                    condition_signals = expr_info.get('involvedSignals', [])
                    all_driving_signals.update(condition_signals)
            
            metrics['dependency_breadth'] = len(all_driving_signals)
        except:
            metrics['dependency_breadth'] = 0
        
        return metrics

    def calculate_functional_metrics(self, var_name):
        """计算功能性指标"""
        var_info = self.data[var_name]
        metrics = {}
        
        # 1. 驱动输出重要性
        metrics['drives_output'] = 1 if var_info.get('drivesOutput', False) else 0
        
        # 2. 控制变量重要性
        metrics['is_control_variable'] = 1 if var_info.get('isControlVariable', False) else 0
        
        # 3. 赋值频率
        metrics['assignment_frequency'] = var_info.get('assignmentCount', 0)
        
        # 4. 端口重要性
        direction = var_info.get('direction', '')
        metrics['is_input'] = 1 if direction == 'input' else 0
        metrics['is_output'] = 1 if direction == 'output' else 0
        metrics['is_inout'] = 1 if direction == 'inout' else 0
        
        # 5. 数据位宽
        metrics['bit_width'] = var_info.get('bitWidth', 1)
        
        return metrics

    def calculate_reliability_metrics(self, var_name):
        """计算可靠性相关指标"""
        metrics = {}
        var_info = self.data[var_name]
        
        # 1. 条件稳定性 (条件中参数 vs 信号的比例)
        total_conditions = 0
        stable_conditions = 0
        
        for assignment in var_info.get('assignments', []):
            for clause in assignment.get('path', []):
                expr_info = clause.get('expr', {})
                params = set(expr_info.get('involvedParameters', []))
                signals = set(expr_info.get('involvedSignals', []))
                
                total_conditions += 1
                if len(params) > len(signals):
                    stable_conditions += 1
        
        metrics['condition_stability'] = stable_conditions / total_conditions if total_conditions > 0 else 1.0
        
        # 2. 复位敏感性
        assignments = var_info.get('assignments', [])
        reset_related = 0
        for assignment in assignments:
            driving_signals = assignment.get('drivingSignals', [])
            if any('reset' in signal.lower() or 'rst' in signal.lower() for signal in driving_signals):
                reset_related += 1
        
        metrics['reset_sensitivity'] = reset_related / len(assignments) if assignments else 0.0
        
        return metrics

    def normalize_scores(self):
        """对评分进行归一化处理"""
        print("Normalizing scores...")
        
        if not self.variable_scores:
            print("No scores to normalize.")
            return
        
        # 收集所有指标的数值
        all_metrics = defaultdict(list)
        for scores in self.variable_scores.values():
            for metric, value in scores.items():
                # 跳过布尔值和字符串类型的指标
                if isinstance(value, (bool, str)):
                    continue
                all_metrics[metric].append(value)
        
        # 对每个数值型指标进行min-max归一化
        normalized_scores = {}
        for var_name, scores in self.variable_scores.items():
            normalized = {}
            for metric, value in scores.items():
                # 保持布尔值和字符串类型不变
                if isinstance(value, (bool, str)):
                    normalized[metric] = value
                    continue
                    
                if metric in all_metrics and all_metrics[metric]:
                    values = all_metrics[metric]
                    min_val = min(values)
                    max_val = max(values)
                    
                    if max_val > min_val:
                        normalized[metric] = (value - min_val) / (max_val - min_val)
                    else:
                        normalized[metric] = 0.5  # 所有值相等时取中值
                else:
                    normalized[metric] = value
            
            # 保存原始值和归一化值
            normalized_scores[var_name] = {
                **normalized,  # 归一化值
                '_raw_scores': scores  # 原始值
            }
        
        self.variable_scores = normalized_scores
        print("Score normalization completed")
# 使用示例
def main():
    # 初始化增强版评分系统
    scorer = EnhancedVariableScoringSystem('/data/fhj/sva-var/results/ibex_core.json')
    
    # 运行完整分析
    scorer.run_comprehensive_analysis()

if __name__ == "__main__":
    main()