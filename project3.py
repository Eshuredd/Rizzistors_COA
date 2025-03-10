import numpy as np
import matplotlib.pyplot as plt
import re  # Import regex module for instruction validation


class Memory:
    def __init__(self, size=4096):
        self.memory = [0] * size

    def write(self, address, value):
        address = address // 4
        print("accessing write", address)
        if 0 <= address < len(self.memory):
            self.memory[address] = value
            print("accessing write done", self.memory[address])
        else:
            raise ValueError("Memory address out of bounds")

    def read(self, address):
        address = address // 4
        print("accessing read", address)
        if 0 <= address < len(self.memory):
            print("accessing read done", self.memory[address])
            return self.memory[address]
        else:
            raise ValueError("Memory address out of bounds")


class PipeliningCores:
    def __init__(self, core_id, program, text_labels, data_labels, shared_memory, simulator):
        self.core_id = core_id
        self.program = program
        self.text_labels = text_labels
        self.data_labels = data_labels
        self.shared_memory = shared_memory
        self.simulator = simulator
        self.registers = [0] * 32
        self.registers[31] = self.core_id
        self.source_active = [0] * 32
        self.pc = 0
        self.stall_count = 0
        self.IF = None
        self.ID = None
        self.EX = None
        self.MEM = None
        self.WB = None
        self.execute_decode = True
        self.TimeExecRem = 0
        self.TimeMemeRem = 0
        self.finished = False
        self.active_registers = [False] * 32  # Track active status of registers
        self.pipeline_stages = {
            'IF': None,
            'ID': None,
            'EX': None,
            'MEM': None,
            'WB': None
        }
        self.instructions = []
        self.clock_cycles = 0
        self.execute_active = True
        self.memory_active = True
        self.latency = {
            "add":1,
            "sub": 1,
            "mul": 1,
            "addi": 1,
            "jalr": 1,
            "slli": 1,
            "lw": 1,
            "sw": 1,
            "la": 1,
            "jal": 1,
            "j": 1,
            "beq": 1,
            "bne": 1,
            "bge": 1
        }
        self.stall_counter = 0
        self.data_forwarding = False
        self.if_aval = True

    def is_finished(self):
        return self.finished
    def forw_EX(self,dest_id) -> any:
        #print("checking in ex",dest_id)
        parts=self.MEM
        #print(parts)
        if not parts:
            return "False"
        if dest_id == int(parts[1][1:]) and parts[0] not in ["lw","la"]:
            data=parts[2]
            return data
        elif dest_id == int(parts[1][1:]) and parts[0] in ["lw","la"]:
            #print("inside exr but lw")
            return "stall"
        else:
            return "False"

    def forw_ME(self,dest_id) -> any:
        #print("checking in me",dest_id)
        parts=self.WB_register
        if not parts:
            return "False"        
        if dest_id == int(parts[1][1:]):
            data=parts[2]
            return data
        else:
            return "False" 
        
    def data_forward(self,dest_id) -> any:
        if self.data_forwarding == False:
            return "False"
        in_exr = self.forw_EX(dest_id)
        if in_exr == "stall":
            return "False"
        elif in_exr == "False":
            in_mer = self.forw_ME(dest_id)
            if in_mer == "False":
                return "False"
            else:
                return in_mer
        else:
            return in_exr   

    def advance_pipeline(self):
        print(f"Current PC: {self.pc}, Pipeline state: IF={self.IF}, ID={self.ID}, EX={self.EX}, MEM={self.MEM}, WB={self.WB}")

        # Create temporary variables to hold new stage values
        new_WB = None
        new_MEM = None
        new_EX = None
        new_ID = None
        new_IF = None

        # Process each stage in reverse order to prevent overwriting

        # Write Back Stage
        if self.WB:
            print(f"Writing back: {self.WB}")
            self.write_back(self.WB)

        # Memory Access Stage - results go to WB
        if self.MEM:
            print(f"Memory access: {self.MEM}")
            new_WB = self.memory_access(self.MEM)

        # Execute Stage - results go to MEM
        if self.EX:
            print(f"Executing: {self.EX}")
            new_MEM = self.execute(self.EX)

        # Instruction Decode Stage - results go to EX
        if self.ID:
            print(f"Decoding: {self.ID}")
            new_EX = self.decode(self.ID)

        self.WB = new_WB
        self.MEM = new_MEM
        self.EX = new_EX

        if self.if_aval == False:
            return
        self.if_aval = False

        # Instruction Fetch Stage - results go to ID
        if self.IF is None and self.pc // 4 < len(self.program) and self.pc >= 0:
            new_IF = self.program[self.pc // 4]
            print(f"Fetched: {new_IF} at PC: {self.pc}")
            new_ID = new_IF  # Move to ID stage immediately
            self.pc += 4     # Increment PC

        # Update all stages with new values

        # Only update ID if we didn't just execute a branch/jump
        if new_EX and (isinstance(new_EX, tuple) and new_EX[0]):
            self.ID = new_ID
            self.IF = None
        elif self.ID is None:  # If ID is empty, fill it
            self.ID = new_ID
            self.IF = None

        # Check if we're finished
        if self.pc // 4 >= len(self.program) and not self.IF and not self.ID and not self.EX and not self.MEM and not self.WB:
            print("Execution complete!")
            self.finished = True

    def write_back(self, instruction):
        opcode = instruction[0]
        if opcode in ["add", "sub", "mul", "addi", "slli", "lw"]:
            opcode, rd, result = instruction
            self.registers[int(rd[1:])] = result
            self.source_active[int(rd[1:])] -= 1
        elif opcode == "la":
            opcode, rd = instruction
            print("dec la count", instruction)
            self.source_active[int(rd[1:])] -= 1

    def memory_access(self, instruction):
        try:
            opcode, rd, address = instruction
            if opcode == "lw":
                value = self.shared_memory.read(address)
                return (opcode, rd, value)
            elif opcode == "sw":
                print("printing ins at sw", instruction)
                value = self.registers[int(rd[1:])]
                self.shared_memory.write(address, value)
            return instruction
        except ValueError as e:
            print(f"Error in memory_access: {e}, instruction: {instruction}")
            return None

    def execute(self, instruction):
        if instruction is None:
            print("Error: Attempted to execute a None instruction")
            return None
        
        if self.TimeExecRem >0:
            self.stall_count+=1
            self.TimeExecRem -=1
            return

        opcode, *operands = instruction
        print(f"Executing instruction with opcode: {opcode}, operands: {operands}")

        if opcode == "add":
            rd, rs1, rs2 = operands
            result = self.registers[int(rs1)] + self.registers[int(rs2)]
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "sub":
            rd, rs1, rs2 = operands
            result = self.registers[int(rs1)] - self.registers[int(rs2)]
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "mul":
            rd, rs1, rs2 = operands
            result = self.registers[int(rs1)] * self.registers[int(rs2)]
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "addi":
            rd, rs1, imm = operands
            result = self.registers[int(rs1)] + int(imm)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "slli":
            rd, rs1, imm = operands
            result = self.registers[int(rs1)] << int(imm)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "la":
            rd, label = operands
            if label in self.data_labels:
                address = self.data_labels[label]
                self.registers[int(rd)] = address  # Directly update register
                print(f"Set register {rd} to address {address}")
            else:
                print(f"Label {label} not found in data_labels")
            self.execute_decode = True
            return (opcode, rd)
        elif opcode == "bge":
            rs1, rs2, label = operands
            val1 = self.registers[int(rs1)]
            val2 = self.registers[int(rs2)]
            print(f"Comparing {rs1}={val1} >= {rs2}={val2}")
            if val1 >= val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    print(f"Branch taken to {label}, PC changing from {self.pc} to {new_pc}")
                    # Important: Set PC correctly and flush pipeline
                    self.pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    print(f"Label {label} not found in text_labels")
            else:
                print(f"Branch not taken: {val1} < {val2}")
            self.execute_decode = True
            return None
        elif opcode == "bne":
            rs1, rs2, label = operands
            val1 = self.registers[int(rs1)]
            val2 = self.registers[int(rs2)]
            print(f"Comparing {rs1}={val1} != {rs2}={val2}")
            if val1 >= val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    print(f"Branch taken to {label}, PC changing from {self.pc} to {new_pc}")
                    # Important: Set PC correctly and flush pipeline
                    self.pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    print(f"Label {label} not found in text_labels")
            else:
                print(f"Branch not taken: {val1} < {val2}")
            self.execute_decode = True
            return None
        elif opcode == "blt":
            rs1, rs2, label = operands
            val1 = self.registers[int(rs1)]
            val2 = self.registers[int(rs2)]
            print(f"Comparing {rs1}={val1} < {rs2}={val2}")
            if val1 < val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    print(f"Branch taken to {label}, PC changing from {self.pc} to {new_pc}")
                    # Important: Set PC correctly and flush pipeline
                    self.pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    print(f"Label {label} not found in text_labels")
            else:
                print(f"Branch not taken: {val1} >= {val2}")
            self.execute_decode = True
            return None
        elif opcode == "j":
            label = operands[0]
            if label in self.text_labels:
                new_pc = self.text_labels[label]
                print(f"Jump to {label}, PC changing from {self.pc} to {new_pc}")
                # Important: Set PC correctly and flush pipeline
                self.pc = new_pc * 4
                self.IF = None
                self.ID = None
                self.if_aval = True
            else:
                print(f"Label {label} not found in text_labels")
            self.execute_decode = True
            return None
        elif opcode in ["lw", "sw"]:
            rd, offset, rs1 = operands
            print("printing before", operands)
            print(self.registers[int(rs1)])
            address = self.registers[int(rs1)] + int(offset)
            print(address)
            self.execute_decode = True
            return (opcode, rd, address)
        else:
            print(f"Unknown opcode during execute: {opcode}")
            
            return None

    def decode(self, instruction):
        if instruction is None:
            print("Error: Attempted to decode a None instruction")
            return None

        parts = instruction.split()
        if not parts:
            print("Error: Empty instruction parts")
            return None

        if self.execute_decode == False:
            return

        opcode = parts[0]
        print(f"Decoding instruction with opcode: {opcode}")

        def get_register_id(reg):
            return int(reg[1:])  # Extract register number from "xN" format

        if opcode in ["add", "sub", "mul"]:
            if len(parts) < 4:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None
            
            source_1 = get_register_id(parts[2])
            source_2 = get_register_id(parts[3])
            dest = get_register_id(parts[1])

            rs1_data=None
            rs2_data=None 
            stall=False
            # Check for hazards, skipping "CID"
            if (source_1 is not None and self.source_active[source_1] > 0) or \
               (source_2 is not None and self.source_active[source_2] > 0):
                if source_1 is not None and self.source_active[source_1] > 0:
                    sr1_from_forw=self.data_forward(source_1)
                    if sr1_from_forw == "False":
                        stall=True
                    else:
                        rs1_data = sr1_from_forw
                if source_2 is not None and self.source_active[source_2] > 0:
                    sr2_from_forw=self.data_forward(source_2)
                    if sr2_from_forw == "False":
                        stall=True
                    else:
                        rs1_data = sr2_from_forw                        

            if stall:
                self.stall_count+=1
                return
            
            if rs1_data is not None:
                parts[2] = rs1_data
            else:
                parts[2] = self.registers[int(parts[2][1:])]

            if rs2_data is not None:
                parts[3]=rs2_data
            else:
                parts[3] = self.registers[int(parts[3][1:])]

            self.source_active[dest] += 1
            if opcode in ["lw","sw"]:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            
            return (opcode, parts[1], parts[2], parts[3])


        elif opcode == "addi":
            if len(parts) < 4:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None
            
            rs1_data=None
            stall = False
            source_1 = get_register_id(parts[2])
            dest = get_register_id(parts[1])

            if source_1 is not None and self.source_active[source_1] > 0:
                if source_1 is not None and self.source_active[source_1] > 0:
                    sr1_from_forw=self.data_forward(source_1)
                    if sr1_from_forw == "False":
                        stall=True
                    else:
                        rs1_data = sr1_from_forw
                return
            self.source_active[dest] += 1
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            if stall:
                self.stall_count+=1
                return
            
            if rs1_data is not None:
                parts[2] = rs1_data
            else:
                parts[2] = self.registers[int(parts[2][1:])]
            return ("addi", parts[1], parts[2], parts[3])

        elif opcode in ["lw"]:
            if len(parts) < 3:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None
            rs1_data=None
            stall = False
            rd = parts[1]
            offset_base = parts[2].split('(')
            offset = offset_base[0]
            rs1 = offset_base[1].rstrip(')')
            dest = get_register_id(rd)
            source_1 = get_register_id(rs1)

            if source_1 is not None and self.source_active[source_1] > 0:
                self.stall_count += 1
                return

            self.source_active[dest] += 1
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            if stall:
                self.stall_count+=1
                return
            
            if rs1_data is not None:
                parts[1] = rs1_data
            else:
                rs1 = self.registers[int(rs1[1:])]
            return (opcode, rd, offset, rs1)

        elif opcode in ["sw"]:
            if len(parts) < 3:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None
            rs1_data=None
            rs2_data=None 
            stall = False
            rd = parts[1]
            offset_base = parts[2].split('(')
            offset = offset_base[0]
            rs1 = offset_base[1].rstrip(')')
            source_1 = get_register_id(rs1)
            source_2 = get_register_id(rd)

            if (source_1 is not None and self.source_active[source_1] > 0) or \
               (source_2 is not None and self.source_active[source_2] > 0):
                self.stall_count += 1
                return
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            if stall:
                self.stall_count+=1
                return
            
            if rs1_data is not None:
                parts[1] = rs1_data
            else:
                parts[1] = self.registers[int(parts[1][1:])]

            if rs2_data is not None:
                rs1 = rs2_data
            else:
                rs1 = self.registers[int(rs1[1:])]
            return (opcode, rd, offset, rs1)

        elif opcode in ["bge", "blt", "bne"]:
            if len(parts) < 4:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None

            source_1 = get_register_id(parts[1])
            source_2 = get_register_id(parts[2])
            stall = False
            rs1_data = None
            rs2_data = None

            if (source_1 is not None and self.source_active[source_1] > 0) or \
               (source_2 is not None and self.source_active[source_2] > 0):
                self.stall_count += 1
                return
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            
            if stall:
                self.stall_count+=1
                return
            
            if rs1_data is not None:
                parts[1] = rs1_data
            else:
                parts[1] = self.registers[int(parts[1][1:])]

            if rs2_data is not None:
                parts[2]=rs2_data
            else:
                parts[2] = self.registers[int(parts[2][1:])]
            return (opcode, parts[1], parts[2], parts[3])

        elif opcode == "la":
            if len(parts) < 3:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None

            dest = get_register_id(parts[1])
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            
            return (opcode, parts[1], parts[2])

        elif opcode == "j":
            if len(parts) < 2:
                print(f"Error: Incomplete instruction for {opcode}: {parts}")
                return None
            if opcode in []:
                self.TimeExecRem=0
            elif opcode in self.latency:
                self.TimeExecRem=self.latency[opcode]-1
            else:
                self.TimeExecRem=0
            self.execute_decode = False
            self.if_aval = True
            return (opcode, parts[1])

        
        print(f"Unknown opcode during decode: {opcode}")
        return None

    def update_register(self, index, value):
        if index != 0:  # Prevent modifying register 0
            self.registers[index] = value

    def fetch_register(self, index) -> any:
        if index.startswith("x"):
            reg_index = int(index[1:])
            return self.registers[reg_index]
        elif index == "cid":
            return self.core_id
        else:
            raise ValueError("Invalid register index")


class PipelinedSimulator:
    def __init__(self, filename):
        self.text_labels = {}
        self.data_labels = {}
        self.data_section = []  # List of (address, value)
        self.program = self.load_program(filename)
        self.pc = 0  # Global PC for shared fetch
        self.current_instruction = None  # Shared fetch buffer
        self.branch_taken = False  # Flag used when a branch is taken
        shared_memory = Memory()  # Shared memory instance for all cores
        self.cores = [PipeliningCores(i, self.program, self.text_labels, self.data_labels, shared_memory, self)
                      for i in range(4)]
        # Write data to the shared memory (only once).
        for addr, value in self.data_section:
            shared_memory.write(addr, value)
        self.clock = 0

    def load_program(self, filename):
        instructions = []
        data_section = []
        in_data = False
        in_text = False
        data_ptr = 0  # word address pointer
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().split('#')[0].strip()
                if not line:
                    continue
                if line.startswith(".data"):
                    in_data = True
                    in_text = False
                    continue
                elif line.startswith(".text"):
                    in_text = True
                    in_data = False
                    continue
                if in_data:
                    if ":" in line:
                        label_part, rest = line.split(":", 1)
                        label = label_part.strip()
                        tokens = rest.strip().split()
                        if tokens and tokens[0] == ".word":
                            self.data_labels[label] = data_ptr * 4
                            for word in tokens[1:]:
                                data_section.append((data_ptr, int(word)))
                                data_ptr += 1
                    else:
                        tokens = line.split()
                        if tokens and tokens[0] == ".word":
                            for word in tokens[1:]:
                                data_section.append((data_ptr, int(word)))
                                data_ptr += 1
                elif in_text:
                    if ":" in line:
                        if line.endswith(":"):
                            label = line.replace(":", "").strip()
                            self.text_labels[label] = len(instructions)
                        else:
                            label, instr = line.split(":", 1)
                            label = label.strip()
                            self.text_labels[label] = len(instructions)
                            if instr.strip():
                                instructions.append(instr.strip())
                    else:
                        instructions.append(line)
        self.data_section = data_section
        return instructions

    def run(self):
        cycle = 0
        max_cycles = 300  # Prevent infinite loops
        
        while True:
            all_finished = True
            
            for i in range(len(self.cores)):
                if not self.cores[i].is_finished():
                    all_finished = False
                    self.cores[i].advance_pipeline()
            
            if all_finished or cycle >= max_cycles:
                break
                
            cycle += 1
        print(f"\nSimulation completed in {cycle} cycles")
        
        # Print final register state
        print("\nFinal Register State:")
        for i in range(len(self.cores)):
            print(f"Core {i} registers:")
            for j in range(32):  # Loop through all 32 registers
                if self.cores[i].registers[j] != 0:
                    print(f"x{j}: {self.cores[i].registers[j]}")

    def is_finished(self):
        return all(core.finished for core in self.cores)

    def print_stall_summary(self):
        total_stalls = 0
        for core in self.cores:
            print(f"Core {core.core_id} - Total stalls: {core.stall_count}")
            total_stalls += core.stall_count
        print(f"Overall stalls across all cores: {total_stalls}")

    def display(self):
        """
        Display function showing register state in terminal and both memory and register state in visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not installed. Cannot generate visualizations.")
            print("Install with: pip install matplotlib")
            return
        
        # Text-based output - only registers, no memory
        print(f"\n===== Simulation Status (Cycle {self.clock}) =====")
        
        print("Core Registers:")
        for i, core in enumerate(self.cores):
            if hasattr(core, 'registers'):
                # Only show non-zero registers to reduce clutter
                non_zero_regs = [(j, val) for j, val in enumerate(core.registers) if val != 0]
                if non_zero_regs:
                    print(f"Core {i} (non-zero registers):")
                    for reg_idx, reg_val in non_zero_regs:
                        print(f"  x{reg_idx}: {reg_val}")
                else:
                    print(f"Core {i}: All registers are zero")
            else:
                print(f"Core {i}: No registers found")
        
        # Create a 2x2 grid of subplots for memory visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        cores_to_display = min(4, len(self.cores))
        
        # Plot memory state for each core
        for i in range(cores_to_display):
            ax = axes[i // 2, i % 2]  # Select the appropriate subplot
            core = self.cores[i]
            
            # Access shared memory (as in the original code)
            if hasattr(core, 'shared_memory') and hasattr(core.shared_memory, 'memory'):
                if isinstance(core.shared_memory.memory, list):
                    # Take first 128 elements or fewer if list is shorter
                    memory_size = min(128, len(core.shared_memory.memory))
                    memory_data = core.shared_memory.memory[:memory_size]
                    memory_reshaped = np.array(memory_data).reshape(-1, 8)  # Display in a 2D grid
                elif isinstance(core.shared_memory.memory, dict):
                    # For dictionary memory, create array of zeros and fill with values
                    memory_array = np.zeros(128)
                    for addr, val in core.shared_memory.memory.items():
                        if 0 <= addr < 128:
                            memory_array[addr] = val
                    memory_reshaped = memory_array.reshape(-1, 8)
                else:
                    # Default empty array if memory format unknown
                    memory_reshaped = np.zeros((16, 8))
            else:
                memory_reshaped = np.zeros((16, 8))
            
            # Display the heatmap
            im = ax.imshow(memory_reshaped, cmap="Reds")
            
            # Add the actual values as text
            for row in range(memory_reshaped.shape[0]):
                for col in range(8):
                    value = memory_reshaped[row, col]
                    ax.text(col, row, str(int(value)), ha="center", va="center", 
                            color="black" if value < 50 else "white")
            
            ax.set_title(f"Core {i} Memory State (First 128 Words)")
            ax.set_xlabel("Memory Offset")
            ax.set_ylabel("Row Index")
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # Create a separate figure for register visualization
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        
        # Prepare register data for all cores
        register_data = np.zeros((cores_to_display, 32))
        for i in range(cores_to_display):
            if i < len(self.cores) and hasattr(self.cores[i], 'registers'):
                register_data[i] = np.array(self.cores[i].registers[:32])
        
        # Display register heatmap
        im2 = ax2.imshow(register_data, cmap="Blues")
        
        # Add text values
        for i in range(cores_to_display):
            for j in range(32):
                value = int(register_data[i, j])
                ax2.text(j, i, str(value), ha="center", va="center", 
                        color="black" if value < 50 else "white")
        
        ax2.set_title("Registers State for Each Core")
        ax2.set_xlabel("Register Index (X0-X31)")
        ax2.set_ylabel("Core ID")
        ax2.set_xticks(range(32))
        ax2.set_xticklabels([f"X{i}" for i in range(32)])
        ax2.set_yticks(range(cores_to_display))
        ax2.set_yticklabels([f"Core {i}" for i in range(cores_to_display)])
        
        # Add colorbar
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        # Show all figures
        try:
            plt.show()
        except Exception as e:
            print(f"Error displaying plots: {e}")
        
        print(f"Current clock cycle: {self.clock}")

# Main execution
if __name__ == "__main__":
    sim = PipelinedSimulator('BubbleSort.asm')
    sim.run()
    sim.print_stall_summary()
    sim.display()
