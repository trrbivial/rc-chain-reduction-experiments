import subprocess
import sys
import os


def run_command(command, use_sudo=False, cwd=None):
    if use_sudo:
        command.insert(0, 'sudo')

    # 存储 stdout 和 stderr 的内容
    stdout_data = []
    stderr_data = []

    # 使用 Popen 来启动进程，并将 stdout 和 stderr 直接输出到屏幕
    with subprocess.Popen(command,
                          text=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          cwd=cwd) as process:
        # 实时输出 stdout 并捕获它
        for line in process.stdout:
            print(line, end='')  # 实时输出 stdout
            stdout_data.append(line)  # 保存 stdout 内容

        # 实时输出 stderr 并捕获它
        for line in process.stderr:
            print(line, end='', file=sys.stderr)  # 实时输出 stderr
            stderr_data.append(line)  # 保存 stderr 内容

        # 等待命令完成并获取返回码
        returncode = process.wait()

    # 返回退出码，以及捕获的 stdout 和 stderr 内容
    return returncode, ''.join(stdout_data), ''.join(stderr_data)


# Step 1: Run ./compile_linux.sh with sudo
print("Compiling...")
code, out, err = run_command(['./compile_linux.sh'], use_sudo=True)
if code != 0:
    print("Compilation failed!")
    sys.exit(1)

print("Compilation successful!")

# Step 2: Check ngspice path
print("Checking ngspice path...")
code, out, err = run_command(['which', 'ngspice'])
if code != 0 or out.strip() != '/usr/local/bin/ngspice':
    print("Error: ngspice path is incorrect or not found!")
    sys.exit(1)

print("ngspice path is correct.")

# Step 3: Run make check in release directory
release_dir = "release"
print("Running make check in release directory...")
code, out, err = run_command(['make', 'check'], use_sudo=True, cwd=release_dir)

# Save output to log file
log_file = "make_check.log"
with open(log_file, "w") as f:
    f.write(out)
    f.write(err)

print(f"Output saved to {log_file}")
