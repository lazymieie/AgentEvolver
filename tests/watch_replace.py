#!/usr/bin/env python3
"""
智能JSONL监控脚本 - 检测到错误后等待确认再修复
"""

import os
import sys
import time
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SmartJSONLHandler(FileSystemEventHandler):
    def __init__(self, file_path, check_script, fix_script):
        """
        初始化监控处理器
        
        参数:
            file_path: 监控的JSONL文件路径
            check_script: 检查脚本路径 (tests/test.py)
            fix_script: 修复脚本路径 (tests/test_replace.py)
        """
        self.file_path = file_path
        self.check_script = check_script
        self.fix_script = fix_script
        
        # 错误跟踪：记录每行错误出现的次数
        self.error_counter = defaultdict(int)  # {line_number: count}
        
        # 配置参数
        self.debounce_time = 2.0  # 防抖时间（秒）
        self.trigger_threshold = 3  # 触发修复的错误次数阈值
        self.error_reset_time = 30  # 错误计数器重置时间（秒）
        self.last_check_time = 0
        self.last_file_size = 0
        
        print(f"🚀 智能JSONL监控已启动")
        print(f"📁 监控文件: {file_path}")
        print(f"🔍 检查脚本: {check_script}")
        print(f"🔧 修复脚本: {fix_script}")
        print(f"⚡ 触发阈值: 连续{self.trigger_threshold}次检测到相同错误")
        print(f"⏱️  防抖时间: {self.debounce_time}秒")
        print("-" * 50)
    
    def on_modified(self, event):
        """文件被修改时触发"""
        if event.src_path == self.file_path:
            current_time = time.time()
            
            # 1. 防抖检查
            if current_time - self.last_check_time < self.debounce_time:
                return
            
            # 2. 文件大小检查（避免处理空文件或微小变动）
            try:
                current_size = os.path.getsize(self.file_path)
                if abs(current_size - self.last_file_size) < 10:  # 变化小于10字节
                    return
                self.last_file_size = current_size
            except:
                pass
            
            self.last_check_time = current_time
            print(f"\n[{time.strftime('%H:%M:%S')}] 🔄 检测到文件变化，开始检查...")
            self.run_check()
    
    def on_created(self, event):
        """文件被创建时触发"""
        if event.src_path == self.file_path:
            print(f"\n[{time.strftime('%H:%M:%S')}] 📄 检测到新文件，开始检查...")
            self.run_check()
    
    def run_check(self):
        """运行检查脚本并分析结果"""
        try:
            # 运行检查脚本
            result = subprocess.run(
                [sys.executable, self.check_script],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print(f"❌ 检查脚本执行失败: {result.stderr[:200]}")
                return
            
            # 解析检查结果
            current_errors = self.parse_check_output(result.stdout)
            
            if not current_errors:
                print("✅ 检查完成：没有发现错误行")
                # 清理旧的错误计数（如果文件已经修复）
                self.cleanup_old_errors()
                return
            
            # 更新错误计数器
            self.update_error_counter(current_errors)
            
            # 显示当前错误状态
            self.print_error_status(current_errors)
            
            # 检查是否需要触发修复
            if self.should_trigger_fix():
                print(f"\n⚠️  达到触发阈值，开始执行修复...")
                self.run_fix()
            
        except subprocess.TimeoutExpired:
            print("⏰ 检查脚本执行超时")
        except Exception as e:
            print(f"❌ 检查过程出错: {e}")
    
    def parse_check_output(self, output):
        """
        解析检查脚本的输出，提取错误行信息
        
        返回: [(line_num, error_msg), ...]
        """
        errors = []
        lines = output.strip().split('\n')
        
        # 查找错误行
        for line in lines:
            line = line.strip()
            if line.startswith('[line '):
                # 解析形如 "[line 123] Expecting property name enclosed in double quotes"
                parts = line.split('] ', 1)
                if len(parts) == 2:
                    line_info = parts[0][6:]  # 去掉 "[line "
                    line_num = int(line_info.split()[0])
                    error_msg = parts[1].strip()
                    errors.append((line_num, error_msg))
        
        return errors
    
    def update_error_counter(self, current_errors):
        """更新错误计数器"""
        current_time = time.time()
        
        # 1. 增加当前错误的计数
        current_error_lines = {line_num for line_num, _ in current_errors}
        for line_num in current_error_lines:
            self.error_counter[line_num] += 1
        
        # 2. 清理不在当前错误中的计数器（如果它们超过重置时间）
        to_remove = []
        for line_num, count in self.error_counter.items():
            if line_num not in current_error_lines:
                # 这里可以添加时间判断，但简化版本直接清除
                to_remove.append(line_num)
        
        for line_num in to_remove:
            del self.error_counter[line_num]
    
    def cleanup_old_errors(self):
        """清理旧的错误计数"""
        # 如果文件没有错误，重置所有计数器
        if len(self.error_counter) > 0:
            print(f"🔄 清理{len(self.error_counter)}个旧错误记录")
            self.error_counter.clear()
    
    def print_error_status(self, current_errors):
        """打印当前错误状态"""
        print(f"📊 发现 {len(current_errors)} 个错误行")
        
        for line_num, error_msg in current_errors[:5]:  # 只显示前5个
            count = self.error_counter[line_num]
            print(f"  第 {line_num} 行: {error_msg[:50]}... (连续{count}次)")
        
        if len(current_errors) > 5:
            print(f"  还有 {len(current_errors) - 5} 个错误...")
    
    def should_trigger_fix(self):
        """判断是否需要触发修复"""
        for line_num, count in self.error_counter.items():
            if count >= self.trigger_threshold:
                return True
        return False
    
    def run_fix(self):
        """运行修复脚本"""
        try:
            print("🔧 执行修复脚本...")
            result = subprocess.run(
                [sys.executable, self.fix_script],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ 修复成功完成")
                
                # 显示修复结果
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in 
                          ['bad lines', '已删除', '清理', '修复', 'valid']):
                        print(f"   {line.strip()}")
                
                # 修复后重置错误计数器
                self.error_counter.clear()
                print("🔄 错误计数器已重置")
                
            else:
                print("❌ 修复脚本执行失败")
                if result.stderr:
                    print("错误信息:")
                    for line in result.stderr.split('\n')[-5:]:
                        if line.strip():
                            print(f"   {line.strip()}")
                            
        except subprocess.TimeoutExpired:
            print("⏰ 修复脚本执行超时")
        except Exception as e:
            print(f"❌ 修复过程出错: {e}")
    
    def periodic_cleanup(self):
        """定期清理（可选，可单独线程运行）"""
        while True:
            time.sleep(self.error_reset_time)
            # 可以在这里添加逻辑定期清理过时的错误计数
            pass

def main():
    # 配置文件路径
    jsonl_path = "/gemini/space/gjx/AgentEvolver/local_vector_store/bfcl_multiturn_800.jsonl"
    check_script = "/gemini/space/gjx/AgentEvolver/tests/test.py"  # 你的检查脚本
    fix_script = "/gemini/space/gjx/AgentEvolver/tests/test_replace.py"  # 你的修复脚本
    
    # 检查文件是否存在
    if not os.path.exists(check_script):
        print(f"❌ 错误: 检查脚本不存在: {check_script}")
        return
    
    if not os.path.exists(fix_script):
        print(f"❌ 错误: 修复脚本不存在: {fix_script}")
        return
    
    # 创建事件处理器和观察者
    event_handler = SmartJSONLHandler(jsonl_path, check_script, fix_script)
    observer = Observer()
    
    # 监控文件所在目录
    watch_dir = os.path.dirname(jsonl_path) if os.path.dirname(jsonl_path) else "."
    observer.schedule(event_handler, path=watch_dir, recursive=False)
    
    try:
        observer.start()
        print("📡 监控已启动，等待文件变化...")
        print("   • 按 Ctrl+C 停止监控")
        print("   • 错误需要连续出现3次才会触发修复")
        print("-" * 50)
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 监控已停止")
    except Exception as e:
        print(f"❌ 监控出错: {e}")
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()