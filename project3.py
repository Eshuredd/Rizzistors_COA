import numpy as np
import matplotlib.pyplot as plt


class Cores:
    def __init__(self, cid, memory):
        self.registers = [0] * 32  # 32 general-purpose registers
        self.pc = 0  # Program counter
        self.coreid = cid  # Core identifier
        self.memory = memory

    def execute(self, pgm, mem, labels, data_map, debug_core_id=None):
        if self.pc >= len(pgm):  # Prevent out-of-bounds error
            return 

        parts = pgm[self.pc].split()
        if not parts:
            self.pc += 1
            return

        opcode = parts[0]

        # Debugging: Print current instruction only for the specified core
        if debug_core_id is not None and self.coreid == debug_core_id:
            print(f"Core {self.coreid}: PC = {self.pc}, Instruction = {pgm[self.pc]}")

        if opcode == "ADD" or opcode == "add":
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            rs2 = int(parts[3][1:])
            self.registers[rd] = self.registers[rs1] + self.registers[rs2]

        elif opcode == "SUB" or opcode == "sub":
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            rs2 = int(parts[3][1:])
            self.registers[rd] = self.registers[rs1] - self.registers[rs2]

        elif opcode == "ADDI" or opcode == "addi":
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            imm = int(parts[3])
            self.registers[rd] = self.registers[rs1] + imm

        elif opcode == "LA" or opcode == "la":
            rd = int(parts[1][1:])
            label = parts[2]
            if label in data_map:
                self.registers[rd] = data_map[label]
            else:
                print(f"Error: Label {label} not found in .data")

        elif opcode == "BLT" or opcode == "blt":
            rs1 = int(parts[1][1:])
            rs2 = int(parts[2][1:])
            label = parts[3]
            if debug_core_id is not None and self.coreid == debug_core_id:
                print(f"Core {self.coreid}: BLT {rs1} < {rs2} = {self.registers[rs1] < self.registers[rs2]}")
            if self.registers[rs1] < self.registers[rs2]:
                self.pc = labels[label]  # Jump to label
            else:
                self.pc += 1

        elif opcode == "BGE" or opcode == "bge":
            rs1 = int(parts[1][1:])
            rs2 = int(parts[2][1:])
            label = parts[3]
            if debug_core_id is not None and self.coreid == debug_core_id:
                print(f"Core {self.coreid}: BGE {rs1} >= {rs2} = {self.registers[rs1] >= self.registers[rs2]}")
            if self.registers[rs1] >= self.registers[rs2]:
                self.pc = labels[label]  # Jump to label
            else:
                self.pc += 1

        elif opcode == "LI" or opcode == "li":
            rd = int(parts[1][1:])
            imm = int(parts[2])
            self.registers[rd] = imm

        elif opcode == "J" or opcode == "j":
            label = parts[1]
            self.pc = labels[label]

        elif opcode == "MUL" or opcode == "mul":
            rd = int(parts[1][1:])
            rs1 = int(parts[2][1:])
            rs2 = int(parts[3][1:])
            self.registers[rd] = self.registers[rs1] * self.registers[rs2]

        elif opcode == "LW" or opcode == "lw":
            rd = int(parts[1][1:])  # Destination register
            mem_operand = parts[2].strip()

            # Parse memory operand (e.g., "4(x1)" or "arr")
            if "(" in mem_operand:
                # Handle offset(base) format (e.g., "4(x1)")
                offset_str, rest = mem_operand.split("(")
                rs1 = int(rest.replace(")", "")[1:])  # Base register
                try:
                    offset = int(offset_str)  # Offset value
                except ValueError:
                    # If offset is a label, resolve it using data_map
                    if offset_str in data_map:
                        offset = data_map[offset_str]
                    else:
                        print(f"Error: data label {offset_str} not found!")
                        self.pc += 1
                        return
                address = self.registers[rs1] + offset
            else:
                # Handle direct address or label (e.g., "arr")
                try:
                    address = int(mem_operand)  # Direct address
                except ValueError:
                    # If operand is a label, resolve it using data_map
                    if mem_operand in data_map:
                        address = data_map[mem_operand]
                    else:
                        print(f"Error: data label {mem_operand} not found!")
                        self.pc += 1
                        return

            # Check if address is word-aligned
            if address % 4 != 0:
                print(f"Error: address {address} is not word-aligned!")
                self.pc += 1
                return

            # Check if address is within the core's memory segment
            if self.coreid * 1024 <= address < (self.coreid + 1) * 1024:
                self.registers[rd] = self.memory[address // 4]
            else:
                print(f"Error: cannot access memory at address {address}!")

        elif opcode == "SW" or opcode == "sw":
            rs2 = int(parts[1][1:])
            offset, rs1 = map(int, parts[2][:-1].split('(x'))
            address = self.registers[rs1] + offset
            if 0 <= address < len(mem):
                mem[address // 4] = self.registers[rs2]
            else:
                if debug_core_id is not None and self.coreid == debug_core_id:
                    print(f"Core {self.coreid}: Memory access violation at address {address}")

        elif opcode == "ECALL" or opcode == "ecall":
            a7 = self.registers[17]  # a7 register stores syscall number
            if a7 == 1:
                if debug_core_id is not None and self.coreid == debug_core_id:
                    print(self.registers[10])  # Print integer (a0)
            elif a7 == 4:
                addr = self.registers[10]  # a0 contains memory address of string
                output_string = ""
                while mem[addr] != 0:  # Read until null character
                    output_string += chr(mem[addr])
                    addr += 1
                if debug_core_id is not None and self.coreid == debug_core_id:
                    print(output_string)
            elif a7 == 10:
                if debug_core_id is not None and self.coreid == debug_core_id:
                    print("Simulation exiting...")
                exit(0)

        # Increment PC unless a branch was taken
        if opcode not in ["BEQ", "BLT", "beq", "blt", "J", "j", "BGE", "bge"]:
            self.pc += 1


class Simulator:
    def __init__(self):
        self.memory = [0] * (4096 // 4)  # 1024 words (4 bytes each)
        self.clock = 0
        self.cores = [Cores(i, self.memory) for i in range(4)]  # 4 cores
        self.program = []
        self.labels = {}
        self.data_map = {}

    def load_program(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        in_data_section = False
        data_index = 0  # Start address for .data section
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line == ".data":
                in_data_section = True
                continue
            elif line == ".text":
                in_data_section = False
                continue

            if in_data_section:
                parts = line.split()
                label = parts[0][:-1]
                if parts[1] == ".word":
                    values = [int(x, 10) for x in parts[2:]]
                    self.data_map[label] = data_index
                    for v in values:
                        self.memory[data_index // 4] = v
                        data_index += 4

                elif parts[1] == ".string":
                    string_value = " ".join(parts[2:]).strip('"')
                    self.data_map[label] = data_index
                    for c in string_value:
                        self.memory[data_index] = ord(c)
                        data_index += 1
                    self.memory[data_index] = 0  # Null termination
                    data_index += 1
            else:
                if ':' in line:
                    label, instruction = line.split(':', 1)
                    self.labels[label] = len(self.program)
                    if instruction.strip():
                        self.program.append(instruction.strip())
                else:
                    self.program.append(line)

    def run(self, debug_core_id=None):
        while any(core.pc < len(self.program) for core in self.cores):
            for core in self.cores:
                core.execute(self.program, self.memory, self.labels, self.data_map, debug_core_id)
                self.clock += 1
        sorted_array = self.memory[self.data_map["base"] // 4 : self.data_map["base"] // 4 + 6]
        print(f"Sorted Array: {sorted_array}")

    def display(self):
        print("Core Registers:")
        for core in self.cores:
            print(f"Core {core.coreid}: {core.registers}")
        print("\nMemory:")
        for i, core in enumerate(self.cores):
            print(f"Core {i} Memory:")
            for j in range(0, len(core.memory), 8):
                print(f"Memory[{j}-{j+7}]: {core.memory[j:j+8]}")

        # Create a 2x2 grid of subplots for memory visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot memory state for each core
        for i, core in enumerate(self.cores):
            ax = axes[i // 2, i % 2]  # Select the appropriate subplot
            memory_reshaped = np.array(core.memory[:128]).reshape(-1, 8)  # Display first 128 words in a 2D grid
            ax.imshow(memory_reshaped, cmap="Reds")
            for row in range(memory_reshaped.shape[0]):
                for col in range(8):
                    ax.text(col, row, str(memory_reshaped[row, col]), ha="center", va="center", color="black")
            ax.set_title(f"Core {i} Memory State (First 128 Words)")
            ax.set_xlabel("Memory Offset")
            ax.set_ylabel("Row Index")

        plt.tight_layout()

        # Create a separate figure for register visualization
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        register_data = np.array([core.registers for core in self.cores])
        ax2.imshow(register_data, cmap="Blues")
        for i in range(4):
            for j in range(32):
                ax2.text(j, i, str(self.cores[i].registers[j]), ha="center", va="center", color="black")
        ax2.set_title("Registers State for Each Core")
        ax2.set_xlabel("Register Index (X0-X31)")
        ax2.set_ylabel("Core ID")
        ax2.set_xticks(range(32))
        ax2.set_xticklabels([f"X{i}" for i in range(32)])
        ax2.set_yticks(range(4))
        ax2.set_yticklabels([f"Core {i}" for i in range(4)])

        plt.tight_layout()
        plt.show()
        print(f"{self.clock}")

# Main execution
sim = Simulator()
sim.load_program('BubbleSort.asm')
sim.run(debug_core_id=0)  # Only show processes for Core 0
sim.display()