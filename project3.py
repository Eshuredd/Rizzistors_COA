import numpy as np
import matplotlib.pyplot as plt
import re  # Import regex module for instruction validation


class Memory:
    def __init__(self, size=1024 * 64):
        self.memory = [0] * size

    def write(self, address, value):
        address = address // 4
        if 0 <= address < len(self.memory):
            self.memory[address] = value
        else:
            raise ValueError(f"Memory address {address} out of bounds")

    def read(self, address):
        address = address // 4
        if 0 <= address < len(self.memory):
            value = self.memory[address]
            return value
        else:
            raise ValueError(f"Memory address {address} out of bounds")

class Cache:
    def __init__(self, size, block_size, associativity, access_latency, replacement_policy="LRU"):
        self.size = size  # Cache size in bytes
        self.block_size = block_size  # Block size in bytes
        self.associativity = associativity  # Number of ways
        self.access_latency = access_latency  # Cache access latency in cycles
        self.replacement_policy = replacement_policy
        
        # Calculate cache parameters
        self.num_sets = max(1, size // (block_size * associativity))
        self.offset_bits = int(np.log2(block_size))
        self.index_bits = max(0, int(np.log2(self.num_sets)))
        self.tag_bits = 32 - self.offset_bits - self.index_bits
        
        # Initialize cache structure using dictionaries for better organization
        self.cache_sets = [{} for _ in range(self.num_sets)]  # Each set is a dictionary mapping tags to valid bits
        self.lru_counters = [{} for _ in range(self.num_sets)]  # For LRU replacement
        self.fifo_queues = [[] for _ in range(self.num_sets)]  # For FIFO replacement using lists as queues
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.accesses = 0

    def get_set_index(self, address):
        return (address >> self.offset_bits) & ((1 << self.index_bits) - 1)

    def get_tag(self, address):
        return address >> (self.offset_bits + self.index_bits)

    def get_block_address(self, address):
        return address & ~((1 << self.offset_bits) - 1)

    def access(self, address, is_write=False):
        self.accesses += 1
        set_index = self.get_set_index(address)
        tag = self.get_tag(address)
        
        # Check if block is in cache
        if tag in self.cache_sets[set_index]:
            self.hits += 1
            if self.replacement_policy == "LRU":
                self.lru_counters[set_index][tag] = self.accesses
            return True, self.access_latency
        
        # Cache miss
        self.misses += 1
        return False, self.access_latency

    def replace_block(self, address):
        set_index = self.get_set_index(address)
        tag = self.get_tag(address)
        cache_set = self.cache_sets[set_index]
        
        # If set is not full, add new block
        if len(cache_set) < self.associativity:
            cache_set[tag] = True
            if self.replacement_policy == "LRU":
                self.lru_counters[set_index][tag] = self.accesses
            elif self.replacement_policy == "FIFO":
                self.fifo_queues[set_index].append(tag)
            return
        
        # Set is full, need to replace
        victim_tag = None
        if self.replacement_policy == "LRU":
            # Find least recently used block
            victim_tag = min(self.lru_counters[set_index].items(), key=lambda x: x[1])[0]
            del self.lru_counters[set_index][victim_tag]
        elif self.replacement_policy == "FIFO":
            # Remove oldest block (first in)
            if self.fifo_queues[set_index]:
                victim_tag = self.fifo_queues[set_index].pop(0)
        
        if victim_tag is not None:
            del cache_set[victim_tag]
        
        # Add new block
        cache_set[tag] = True
        if self.replacement_policy == "LRU":
            self.lru_counters[set_index][tag] = self.accesses
        elif self.replacement_policy == "FIFO":
            self.fifo_queues[set_index].append(tag)

    def get_miss_rate(self):
        return self.misses / self.accesses if self.accesses > 0 else 0

    def get_statistics(self):
        return {
            'hits': self.hits,
            'misses': self.misses,
            'accesses': self.accesses,
            'miss_rate': self.get_miss_rate()
        }

    def reset_statistics(self):
        self.hits = 0
        self.misses = 0
        self.accesses = 0

class CacheHierarchy:
    def __init__(self, config):
        # Initialize L1 caches
        self.l1i = Cache(
            config['l1i_size'],
            config['l1_block_size'],
            config['l1_associativity'],
            config['l1_access_latency'],
            config['l1_replacement_policy']
        )
        
        self.l1d = Cache(
            config['l1d_size'],
            config['l1_block_size'],
            config['l1_associativity'],
            config['l1_access_latency'],
            config['l1_replacement_policy']
        )
        
        # Initialize L2 cache
        self.l2 = Cache(
            config['l2_size'],
            config['l2_block_size'],
            config['l2_associativity'],
            config['l2_access_latency'],
            config['l2_replacement_policy']
        )
        
        self.memory_latency = config['memory_latency']
        
    def access_instruction(self, address):
        # Try L1I
        hit, l1_latency = self.l1i.access(address)
        if hit:
            return l1_latency
        
        # L1I miss, try L2
        hit, l2_latency = self.l2.access(address)
        total_latency = l1_latency + l2_latency
        
        if hit:
            # Update L1I
            self.l1i.replace_block(address)
            return total_latency
        
        # L2 miss, access memory
        total_latency += self.memory_latency
        self.l2.replace_block(address)
        self.l1i.replace_block(address)
        return total_latency

    def access_data(self, address, is_write=False):
        # Try L1D
        hit, l1_latency = self.l1d.access(address, is_write)
        if hit:
            return l1_latency
        
        # L1D miss, try L2
        hit, l2_latency = self.l2.access(address)
        total_latency = l1_latency + l2_latency
        
        if hit:
            # Update L1D
            self.l1d.replace_block(address)
            return total_latency
        
        # L2 miss, access memory
        total_latency += self.memory_latency
        self.l2.replace_block(address)
        self.l1d.replace_block(address)
        return total_latency

    def get_statistics(self):
        return {
            'l1i_hits': self.l1i.hits,
            'l1i_misses': self.l1i.misses,
            'l1i_miss_rate': self.l1i.get_miss_rate(),
            'l1d_hits': self.l1d.hits,
            'l1d_misses': self.l1d.misses,
            'l1d_miss_rate': self.l1d.get_miss_rate(),
            'l2_hits': self.l2.hits,
            'l2_misses': self.l2.misses,
            'l2_miss_rate': self.l2.get_miss_rate()
        }

    def reset_statistics(self):
        """Reset statistics for all caches"""
        self.l1i.reset_statistics()
        self.l1d.reset_statistics()
        self.l2.reset_statistics()

    def get_total_accesses(self):
        """Get total number of cache accesses"""
        return (self.l1i.accesses + self.l1d.accesses + self.l2.accesses)

class ScratchpadMemory:
    def __init__(self, size=400, access_latency = 1):  # Default size 400 bytes = 100 words
        self.size = size
        self.memory = [0] * (size // 4)  # We store words (4 bytes each)
        self.access_latency = access_latency # Default access latency (same as L1D cache)
    
    def read(self, address):
        """Read a word from scratchpad memory"""
        word_address = address // 4  # Convert byte address to word address
        if 0 <= word_address < len(self.memory):
            value = self.memory[word_address]
            return value
        else:
            raise ValueError(f"Scratchpad memory access out of bounds at address {address}.")
    
    def write(self, address, value):
        """Write a word to scratchpad memory"""
        word_address = address // 4  # Convert byte address to word address
        if 0 <= word_address < len(self.memory):
            self.memory[word_address] = value
        else:
            raise ValueError(f"Scratchpad memory access out of bounds at address {address}.")
    
    def get_size(self):
        """Return the size of the scratchpad memory in bytes"""
        return self.size
    
    def clear(self):
        """Clear the scratchpad memory"""
        self.memory = [0] * (self.size // 4)

class PipeliningCores:
    def __init__(self, core_id, program, text_labels, data_labels, shared_memory, cache_hierarchy, global_pc):
        self.core_id = core_id
        self.program = program
        self.text_labels = text_labels
        self.data_labels = data_labels
        self.shared_memory = shared_memory
        self.cache_hierarchy = cache_hierarchy
        self.global_pc = global_pc  # Use global PC
        
        # Flag to indicate an exit instruction is in the pipeline
        self.exit_in_pipeline = False
        
        # Initialize scratchpad memory with the same size and latency as L1D cache
        l1d_size = cache_hierarchy.l1d.size
        l1d_latency = cache_hierarchy.l1d.access_latency
        self.scratchpad_memory = ScratchpadMemory(size=l1d_size, access_latency=l1d_latency)
        
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
            "lw_spm": 1, # Add SPM instructions with same latency as L1D
            "sw_spm": 1, # Add SPM instructions with same latency as L1D
            "la": 1,
            "jal": 1,
            "j": 1,
            "beq": 1,
            "bne": 1,
            "bge": 1,
            "sync": 1  # SYNC is a single-cycle operation for signal triggering
        }
        self.stall_counter = 0
        self.data_forwarding = False
        self.if_aval = True
        self.sync_reached = False  # Flag to indicate if SYNC point has been reached
        self.sync_pc = -1  # PC value where SYNC was encountered
        self.sync_count = 0  # Counter for SYNC barriers encountered
        
        # Add a counter for completed instructions
        self.instructions_completed = 0

    def is_finished(self):
        return self.finished
    
    def forw_EX(self, dest_id) -> any:
        parts = self.MEM
        if not parts:
            return "False"
        
        # For standard instructions with destination in second position
        if len(parts) >= 3 and dest_id == int(parts[1][1:]) and parts[0] not in ["lw", "lw_spm", "la", "sw", "sw_spm"]:
            data = parts[2]
            return data
        # For load instructions with dest in second position but result not yet available
        elif len(parts) >= 3 and dest_id == int(parts[1][1:]) and parts[0] in ["lw", "lw_spm", "la"]:
            return "stall"
        else:
            return "False"

    def forw_ME(self, dest_id) -> any:
        parts = self.WB
        if not parts:
            return "False"
            
        # For standard instructions with destination in second position
        if len(parts) >= 3 and dest_id == int(parts[1][1:]):
            data = parts[2]
            return data
        else:
            return "False" 
        
    def data_forward(self, dest_id) -> any:
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
        # If the core is waiting at a SYNC barrier, don't advance the pipeline
        # but still process what's already in the pipeline
        if self.sync_reached and self.IF is None and self.ID is None and self.EX is None and self.MEM is None and self.WB is None:
            # Core is fully stalled at SYNC barrier with empty pipeline
            self.stall_count += 1
            return

        # Create temporary variables to hold new stage values
        new_WB = None
        new_MEM = None
        new_EX = None
        new_ID = None
        new_IF = None

        # Process each stage in reverse order to prevent overwriting

        # Write Back Stage
        if self.WB:
            self.write_back(self.WB)

        # Memory Access Stage - results go to WB
        if self.MEM:
            new_WB = self.memory_access(self.MEM)

        # Execute Stage - results go to MEM
        if self.EX:
            new_MEM = self.execute(self.EX)

        # Instruction Decode Stage - results go to EX
        if self.ID:
            new_EX = self.decode(self.ID)

        # Update pipeline stages
        self.WB = new_WB
        self.MEM = new_MEM
        self.EX = new_EX

        # Only fetch new instructions if not waiting at a SYNC barrier
        if self.sync_reached:
            return

        if self.if_aval == False:
            return
        self.if_aval = False

        # NOTE: Removed the check for add x0, x0, x0 here since we now handle it in the writeback stage

        # Instruction Fetch Stage - results go to ID
        if self.IF is None and self.global_pc // 4 < len(self.program) and self.global_pc >= 0:
            # Access instruction through cache hierarchy
            latency = self.cache_hierarchy.access_instruction(self.global_pc)
            new_IF = self.program[self.global_pc // 4]
            new_ID = new_IF  # Move to ID stage immediately
            self.global_pc += 4     # Increment global PC by 4 bytes (one instruction)
            # Simulate variable latency by stalling if latency > 1
            if latency > 1:
                self.stall_count += latency - 1

        # Update all stages with new values

        # Only update ID if we didn't just execute a branch/jump
        if new_EX and (isinstance(new_EX, tuple) and new_EX[0]):
            self.ID = new_ID
            self.IF = None
        elif self.ID is None:  # If ID is empty, fill it
            self.ID = new_ID
            self.IF = None

        # Check if we're finished - based on program counter and empty pipeline
        if self.global_pc // 4 >= len(self.program) and not self.IF and not self.ID and not self.EX and not self.MEM and not self.WB:
            self.finished = True

    def memory_access(self, instruction):
        try:
            if instruction is None:
                return None
                
            # Check if this is an exit instruction by checking for the special flag
            is_exit_instruction = False
            exit_flag = None
            if isinstance(instruction, tuple) and len(instruction) > 3 and instruction[3] == "exit_instruction":
                is_exit_instruction = True
                exit_flag = instruction[3]
                
            # Special handling for la instruction
            if isinstance(instruction, tuple) and len(instruction) == 2 and instruction[0] == "la":
                # Just pass through la instructions - they were already handled in execute
                return instruction
                
            opcode, rd, address = instruction[:3]
            if opcode == "lw":
                # Access memory through cache hierarchy for load
                latency = self.cache_hierarchy.access_data(address)
                value = self.shared_memory.read(address)
                result_tuple = (opcode, rd, value, latency)
                # Add exit flag if present
                if is_exit_instruction:
                    return result_tuple + (exit_flag,)
                return result_tuple
            elif opcode == "sw":
                # Access memory through cache hierarchy for store
                latency = self.cache_hierarchy.access_data(address, is_write=True)
                value = self.registers[int(rd[1:])]
                self.shared_memory.write(address, value)
                result_tuple = (opcode, rd, address, latency)
                # Add exit flag if present
                if is_exit_instruction:
                    return result_tuple + (exit_flag,)
                return result_tuple
            elif opcode == "lw_spm":
                # Access scratchpad memory for load
                latency = self.scratchpad_memory.access_latency
                value = self.scratchpad_memory.read(address)
                result_tuple = (opcode, rd, value, latency)
                # Add exit flag if present
                if is_exit_instruction:
                    return result_tuple + (exit_flag,)
                return result_tuple
            elif opcode == "sw_spm":
                # Access scratchpad memory for store
                latency = self.scratchpad_memory.access_latency
                
                # Get the value to store - either from the forwarded value or from register
                if len(instruction) > 3 and isinstance(instruction[3], int):
                    # Use forwarded value
                    value = instruction[3]  
                else:
                    # Use register value
                    value = self.registers[int(rd[1:])]
                    
                self.scratchpad_memory.write(address, value)
                result_tuple = (opcode, rd, address, latency)
                # Add exit flag if present
                if is_exit_instruction:
                    return result_tuple + (exit_flag,)
                return result_tuple
            
            # For non-memory instructions, pass them through with the exit flag if present
            if is_exit_instruction:
                return instruction  # The exit flag is already in the instruction
            return instruction
        except Exception as e:
            print(f"Error in memory_access: {e}, instruction: {instruction}")
            return None

    def execute(self, instruction):
        if instruction is None:
            return None
        
        if self.TimeExecRem > 0:
            self.stall_count += 1
            self.TimeExecRem -= 1
            self.EX = instruction
            return None

        opcode, *operands = instruction
        
        if opcode == "add":
            rd, rs1, rs2 = operands
            # Check for exit pattern (add x0, x0, x0)
            if rd == "x0" and int(rs1) == 0 and int(rs2) == 0:
                # Set flag to terminate after this instruction completes writeback
                self.exit_in_pipeline = True
                # Pass a special flag with the instruction to identify it as an exit instruction
                result = int(rs1) + int(rs2)
                self.execute_decode = True
                return (opcode, rd, result, "exit_instruction")
                
            result = int(rs1) + int(rs2)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "sub":
            rd, rs1, rs2 = operands
            result = int(rs1) - int(rs2)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "mul":
            rd, rs1, rs2 = operands
            result = int(rs1) * int(rs2)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "addi":
            rd, rs1, imm = operands
            result = int(rs1) + int(imm)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "slli":
            rd, rs1, imm = operands
            result = int(rs1) << int(imm)
            self.execute_decode = True
            return (opcode, rd, result)
        elif opcode == "la":
            rd, label = operands
            if label in self.data_labels:
                address = self.data_labels[label]
                self.registers[int(rd[1:])] = address  # Directly update register
            self.execute_decode = True
            return (opcode, rd)
        elif opcode == "sync":
            # Record that this core has reached a SYNC instruction
            barrier_id = 0
            if len(operands) > 0:  # If the sync has an ID parameter
                barrier_id = int(operands[0])
                
            # Store the current PC value for this SYNC point
            self.sync_pc = self.global_pc - 4  # The PC has already been incremented
            
            # Set the sync_reached flag to true to indicate we're waiting
            self.sync_reached = True
            self.sync_count = barrier_id
            
            # This is a hardware-compatible implementation that returns the instruction
            # to mark this as a completed operation in the pipeline
            self.execute_decode = True
            return ("sync", barrier_id)
        elif opcode == "beq":
            rs1, rs2, label = operands
            val1 = int(rs1)
            val2 = int(rs2)
            
            # For beq, always take the branch when rs1 == rs2
            if val1 == val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    
                    # Check if we're branching to a termination point
                    if label == "exit" or (new_pc < self.global_pc//4 and self.program[new_pc].startswith("add x0")):
                        self.finished = True
                        return None
                        
                    # IMPORTANT: Make sure to set the PC correctly (in words, not bytes)
                    self.global_pc = new_pc * 4
                    
                    # Only flush IF and ID stages, but leave EX, MEM, and WB intact
                    # This is important to prevent data loss from instructions in later stages
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    # If label not found, terminate to prevent infinite loops
                    self.finished = True
            
            self.execute_decode = True
            return None
        elif opcode == "bne":
            rs1, rs2, label = operands
            val1 = int(rs1)
            val2 = int(rs2)
            if val1 != val2:  # Fixed comparison to != instead of >=
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    
                    # Check if we're branching to a termination point
                    if label == "exit" or (new_pc < self.global_pc//4 and self.program[new_pc].startswith("add x0")):
                        self.finished = True
                        return None
                        
                    # IMPORTANT: Only flush IF and ID stages, preserve MEM and WB
                    self.global_pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    # If label not found, terminate to prevent infinite loops
                    self.finished = True
            
            self.execute_decode = True
            return None
        elif opcode == "bge":
            rs1, rs2, label = operands
            val1 = int(rs1)
            val2 = int(rs2)
            if val1 >= val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    
                    # Check if we're branching to a termination point
                    if label == "exit" or (new_pc < self.global_pc//4 and self.program[new_pc].startswith("add x0")):
                        self.finished = True
                        return None
                    
                    # IMPORTANT: Only flush IF and ID stages, preserve MEM and WB
                    self.global_pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    # If label not found, terminate to prevent infinite loops
                    self.finished = True
            
            self.execute_decode = True
            return None
        elif opcode == "blt":
            rs1, rs2, label = operands
            val1 = int(rs1)
            val2 = int(rs2)
            if val1 < val2:
                if label in self.text_labels:
                    new_pc = self.text_labels[label]
                    
                    # Check if we're branching to a termination point
                    if label == "exit" or (new_pc < self.global_pc//4 and self.program[new_pc].startswith("add x0")):
                        self.finished = True
                        return None
                    
                    # IMPORTANT: Only flush IF and ID stages, preserve MEM and WB
                    self.global_pc = new_pc * 4
                    self.IF = None
                    self.ID = None
                    self.if_aval = True
                else:
                    # If label not found, terminate to prevent infinite loops
                    self.finished = True
            
            self.execute_decode = True
            return None
        elif opcode == "j":
            label = operands[0]
            if label in self.text_labels:
                new_pc = self.text_labels[label]
                
                # Check if we're jumping to a termination point
                if label == "exit" or (new_pc < self.global_pc//4 and self.program[new_pc].startswith("add x0")):
                    self.finished = True
                    return None
                
                # IMPORTANT: Only flush IF and ID stages, preserve MEM and WB
                self.global_pc = new_pc * 4
                self.IF = None
                self.ID = None
                self.if_aval = True
            else:
                # If label not found, terminate to prevent infinite loops
                self.finished = True
            self.execute_decode = True
            return None
        elif opcode in ["lw", "sw"]:
            rd, offset, rs1 = operands
            address = int(rs1) + int(offset)
            self.execute_decode = True
            return (opcode, rd, address)
        elif opcode in ["lw_spm", "sw_spm"]:
            rd, offset, rs1 = operands[:3]
            address = int(rs1) + int(offset)
            
            # Handle forwarded value for sw_spm
            if opcode == "sw_spm" and len(operands) > 3:
                # We have a forwarded value, pass it along
                forwarded_value = operands[3]
                self.execute_decode = True
                return (opcode, rd, address, forwarded_value)
            
            self.execute_decode = True
            return (opcode, rd, address)

        return None

    def decode(self, instruction):
        if instruction is None:
            return None

        parts = instruction.split()
        if not parts:
            return None

        # Clean up the instruction parts to remove commas
        for i in range(len(parts)):
            parts[i] = parts[i].replace(',', '')

        if self.execute_decode == False:
            return self.EX

        opcode = parts[0]

        def get_register_id(reg):
            if reg == "cid":
                return None  # Special case for core ID
            elif reg.startswith("x"):
                try:
                    return int(reg[1:])  # Extract register number from "xN" format
                except ValueError:
                    return None
            else:
                return None

        # Add handling for sync instruction
        if opcode == "sync":
            # SYNC can optionally take a barrier ID parameter
            barrier_id = 0
            if len(parts) > 1:
                try:
                    barrier_id = int(parts[1])
                except ValueError:
                    barrier_id = 0
            
            # Set up execution parameters
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
                
            self.execute_decode = False
            self.if_aval = True
                
            return (opcode, barrier_id)
            
        if opcode in ["add", "sub", "mul"]:
            if len(parts) < 4:
                return None
            
            source_1 = get_register_id(parts[2])
            source_2 = get_register_id(parts[3])
            dest = get_register_id(parts[1])
            
            rs1_data = None
            rs2_data = None 
            stall = False
            # Check for hazards, skipping "CID"
            if (source_1 is not None and self.source_active[source_1] > 0) or \
               (source_2 is not None and self.source_active[source_2] > 0):
                if source_1 is not None and self.source_active[source_1] > 0:
                    sr1_from_forw = self.data_forward(source_1)
                    if sr1_from_forw == "False":
                        stall = True
                    else:
                        rs1_data = sr1_from_forw
                if source_2 is not None and self.source_active[source_2] > 0:
                    sr2_from_forw = self.data_forward(source_2)
                    if sr2_from_forw == "False":
                        stall = True
                    else:
                        rs2_data = sr2_from_forw                        

            if stall:
                self.stall_count += 1
                return
                
            if rs1_data is not None:
                rs1_value = rs1_data
            elif parts[2] == "cid":
                rs1_value = self.core_id
            elif parts[2].startswith("x"):
                rs1_value = self.registers[int(parts[2][1:])]
            else:
                rs1_value = int(parts[2])
            
            if rs2_data is not None:
                rs2_value = rs2_data
            elif parts[3] == "cid":
                rs2_value = self.core_id
            elif parts[3].startswith("x"):
                rs2_value = self.registers[int(parts[3][1:])]
            else:
                rs2_value = int(parts[3])

            self.source_active[dest] += 1
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
            self.execute_decode = False
            self.if_aval = True
            
            return (opcode, parts[1], rs1_value, rs2_value)

        elif opcode == "addi":
            if len(parts) < 4:
                return None
            
            rs1_data = None
            stall = False
            source_1 = get_register_id(parts[2])
            dest = get_register_id(parts[1])

            # Check for hazards
            if source_1 is not None and self.source_active[source_1] > 0:
                sr1_from_forw = self.data_forward(source_1)
                if sr1_from_forw == "False":
                    stall = True
                else:
                    rs1_data = sr1_from_forw
            
            if stall:
                self.stall_count += 1
                return
                
            self.source_active[dest] += 1
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
            self.execute_decode = False
            self.if_aval = True
            
            if rs1_data is not None:
                rs1_value = rs1_data
            elif parts[2] == "cid":
                rs1_value = self.core_id
            elif parts[2].startswith("x"):
                rs1_value = self.registers[int(parts[2][1:])]
            else:
                rs1_value = int(parts[2])
                
            return (opcode, parts[1], rs1_value, parts[3])

        elif opcode in ["lw"]:
            if len(parts) < 3:
                return None
            rs1_data=None
            stall = False
            rd = parts[1]
            
            # Handle memory address format like "offset(register)"
            memory_addr_part = parts[2]
            if "(" in memory_addr_part and ")" in memory_addr_part:
                offset_base = memory_addr_part.split('(')
                offset = offset_base[0]
                rs1 = offset_base[1].rstrip(')')
            else:
                return None
                
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
            elif rs1 == "cid":
                rs1 = self.core_id
            else:
                rs1 = self.registers[int(rs1[1:])]
            return (opcode, rd, offset, rs1)

        elif opcode in ["sw"]:
            if len(parts) < 3:
                return None
            rs1_data=None
            rs2_data=None 
            stall = False
            rd = parts[1]
            
            # Handle memory address format like "offset(register)"
            memory_addr_part = parts[2]
            if "(" in memory_addr_part and ")" in memory_addr_part:
                offset_base = memory_addr_part.split('(')
                offset = offset_base[0]
                rs1 = offset_base[1].rstrip(')')
            else:
                return None
                
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
            elif parts[1] == "cid":
                parts[1] = self.core_id
            else:
                parts[1] = self.registers[int(parts[1][1:])]

            if rs2_data is not None:
                rs1 = rs2_data
            elif rs1 == "cid":
                rs1 = self.core_id
            else:
                rs1 = self.registers[int(rs1[1:])]
            return (opcode, rd, offset, rs1)

        elif opcode in ["beq", "bne", "bge", "blt"]:
            if len(parts) < 4:
                return None
                
            rs1 = parts[1]
            rs2 = parts[2]
            label = parts[3]
            
            source_1 = get_register_id(rs1)
            source_2 = get_register_id(rs2)
            rs1_data = None
            rs2_data = None
            stall = False
            
            # Check for hazards
            if source_1 is not None and self.source_active[source_1] > 0:
                sr1_from_forw = self.data_forward(source_1)
                if sr1_from_forw == "False":
                    stall = True
                else:
                    rs1_data = sr1_from_forw
                    
            if source_2 is not None and self.source_active[source_2] > 0:
                sr2_from_forw = self.data_forward(source_2)
                if sr2_from_forw == "False":
                    stall = True
                else:
                    rs2_data = sr2_from_forw
                    
            if stall:
                self.stall_count += 1
                return
            
            # Get actual register values
            if rs1_data is not None:
                rs1_value = rs1_data
            elif rs1 == "cid":
                rs1_value = self.core_id
            elif rs1.startswith('x'):
                rs1_value = self.registers[int(rs1[1:])]
            else:
                rs1_value = int(rs1)
                
            if rs2_data is not None:
                rs2_value = rs2_data
            elif rs2 == "cid":
                rs2_value = self.core_id
            elif rs2.startswith('x'):
                rs2_value = self.registers[int(rs2[1:])]
            else:
                rs2_value = int(rs2)
                
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
                
            self.execute_decode = False
            self.if_aval = True
            
            return (opcode, rs1_value, rs2_value, label)

        elif opcode == "la":
            if len(parts) < 3:
                return None

            dest = get_register_id(parts[1])
            # Properly increment the source_active counter
            self.source_active[dest] += 1
            
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
                
            self.execute_decode = False
            self.if_aval = True
            
            # Return the appropriate tuple for la instruction
            return (opcode, parts[1], parts[2])

        elif opcode == "j":
            if len(parts) < 2:
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

        elif opcode in ["lw_spm", "sw_spm"]:
            if len(parts) < 3:
                return None
                
            rd = parts[1]
            
            # Handle memory address format like "offset(register)"
            memory_addr_part = parts[2]
            if "(" in memory_addr_part and ")" in memory_addr_part:
                offset_base = memory_addr_part.split('(')
                offset = offset_base[0]
                rs1 = offset_base[1].rstrip(')')
            else:
                return None
                
            rs1_data = None
            rs2_data = None
            stall = False
            
            # Get register IDs and check for hazards
            if opcode == "lw_spm":
                dest = get_register_id(rd)
                source_1 = get_register_id(rs1)
                
                if source_1 is not None and self.source_active[source_1] > 0:
                    # Check if forwarding is possible
                    sr1_from_forw = self.data_forward(source_1)
                    if sr1_from_forw == "False":
                        stall = True
                    else:
                        rs1_data = sr1_from_forw
                    
                self.source_active[dest] += 1
            else:  # sw_spm
                source_1 = get_register_id(rs1)  # Base register
                source_2 = get_register_id(rd)   # Value register
                
                # Check if either source register is being produced by an earlier instruction
                if source_1 is not None and self.source_active[source_1] > 0:
                    # Try to forward the base register
                    sr1_from_forw = self.data_forward(source_1)
                    if sr1_from_forw == "False":
                        stall = True
                    else:
                        rs1_data = sr1_from_forw
                
                if source_2 is not None and self.source_active[source_2] > 0:
                    # Try to forward the value register
                    sr2_from_forw = self.data_forward(source_2)
                    if sr2_from_forw == "False":
                        stall = True
                    else:
                        rs2_data = sr2_from_forw
                
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
                
            self.execute_decode = False
            self.if_aval = True
            
            if stall:
                self.stall_count += 1
                return
            
            # Handle register values for the base address register
            if rs1_data is not None:
                rs1_value = rs1_data
            elif rs1 == "cid":
                rs1_value = self.core_id
            else:
                rs1_value = self.registers[int(rs1[1:])]
            
            # For SW_SPM, we handle the forwarded value in execute stage
            # by passing the value register either as a register name or a forwarded value
            if opcode == "sw_spm" and rs2_data is not None:
                # If we have a forwarded value for source_2, pass it along
                return (opcode, rd, offset, rs1_value, rs2_data)
            else:
                # Normal case - no forwarding for value register
                return (opcode, rd, offset, rs1_value)

        elif opcode == "slli":
            if len(parts) < 4:
                return None
            
            rd = parts[1]
            rs1 = parts[2]
            imm = parts[3]
            
            rs1_data = None
            stall = False
            source_1 = get_register_id(rs1)
            dest = get_register_id(rd)

            if source_1 is not None and self.source_active[source_1] > 0:
                sr1_from_forw = self.data_forward(source_1)
                if sr1_from_forw == "False":
                    stall = True
                else:
                    rs1_data = sr1_from_forw
                
            if stall:
                self.stall_count += 1
                return
            
            self.source_active[dest] += 1
            
            if opcode in self.latency:
                self.TimeExecRem = self.latency[opcode] - 1
            else:
                self.TimeExecRem = 0
                
            self.execute_decode = False
            self.if_aval = True
            
            if rs1_data is not None:
                rs1_value = rs1_data
            elif rs1 == "cid":
                rs1_value = self.core_id
            else:
                rs1_value = self.registers[int(rs1[1:])]
            
            # Shift amount is an immediate value
            imm_value = int(imm)
            
            return (opcode, rd, rs1_value, imm_value)

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

    def access_memory(self, address, is_write=False):
        """Access memory through the cache hierarchy"""
        if is_write:
            return self.cache_hierarchy.access_data(address, is_write=True)
        else:
            return self.cache_hierarchy.access_data(address)

    def access_instruction(self, address):
        """Access instruction through the cache hierarchy"""
        return self.cache_hierarchy.access_instruction(address)

    def write_back(self, instruction):
        if instruction is None:
            return
            
        # Check if this is an exit instruction by looking for the special flag
        is_exit_instruction = False
        if isinstance(instruction, tuple) and len(instruction) > 3 and instruction[3] == "exit_instruction":
            is_exit_instruction = True
            
        opcode = instruction[0]
        # Handle la instruction specially
        if opcode == "la":
            # la instructions have already updated the register in execute
            # Just decrement the source_active counter
            if len(instruction) >= 2:
                rd = instruction[1]
                reg_id = int(rd[1:])
                if reg_id > 0 and reg_id < len(self.source_active):
                    self.source_active[reg_id] -= 1
            
            # Increment instructions completed counter
            self.instructions_completed += 1
            return
            
        # Handle normal register-writing instructions
        if opcode in ["add", "sub", "mul", "addi", "slli", "lw", "lw_spm"]:
            # Unpack the instruction, ignoring latency if present
            if len(instruction) == 4 and not is_exit_instruction:
                opcode, rd, result, _ = instruction  # Ignore latency
            else:
                # Handle regular instruction or exit instruction
                opcode, rd, result = instruction[:3]
            
            # Get the register number
            reg_id = int(rd[1:])
            
            # Update the register (except x0 which is always 0)
            if reg_id != 0:
                self.registers[reg_id] = result
            
            # Decrement the source_active counter for this register
            if reg_id < len(self.source_active):
                self.source_active[reg_id] -= 1
        
        # For store instructions, we also count them as completed
        elif opcode in ["sw", "sw_spm"]:
            # No register value to update, but we still count this as a completed instruction
            pass
            
        # Count branch and jump instructions
        elif opcode in ["beq", "bne", "bge", "blt", "j", "jal", "jalr"]:
            # These have already done their work in the execute stage
            pass
            
        # Count sync instructions 
        elif opcode == "sync":
            # SYNC operations are also counted
            pass
            
        # Increment instructions completed counter
        self.instructions_completed += 1
                
        # After processing the instruction, check if it was an exit instruction
        if is_exit_instruction:
            self.finished = True


class PipelinedSimulator:
    def __init__(self, filename, cache_filename):
        self.text_labels = {}
        self.data_labels = {}
        self.data_section = []  # List of (address, value)
        self.global_pc = 0  # Global PC for shared fetch
        self.current_instruction = None  # Shared fetch buffer
        self.branch_taken = False  # Flag used when a branch is taken
        shared_memory = Memory()  # Shared memory instance for all cores
        self.cache_hierarchy = CacheHierarchy(self.load_cache_config(cache_filename))  # Initialize cache hierarchy
        
        # Create a global scratchpad memory for all cores
        l1d_size = self.cache_hierarchy.l1d.size
        l1d_latency = self.cache_hierarchy.l1d.access_latency
        self.global_scratchpad = ScratchpadMemory(size=l1d_size, access_latency=l1d_latency)
        
        # Synchronization hardware - shared between cores
        self.sync_barrier = {}  # Maps barrier IDs to count of cores that have reached that barrier
        self.active_core_count = 4  # Number of active cores (can be reduced when cores complete)
        
        # Initialize cores with empty program first
        self.cores = [PipeliningCores(i, [], self.text_labels, self.data_labels, shared_memory, self.cache_hierarchy, self.global_pc)
                     for i in range(4)]
        
        # Set the global scratchpad for each core
        for core in self.cores:
            core.scratchpad_memory = self.global_scratchpad
        
        # Load program
        self.program = self.load_program(filename)
        
        # Update each core with the loaded program
        for core in self.cores:
            core.program = self.program
        
        # Write data to the shared memory (only once)
        for addr, value in self.data_section:
            # Convert word address to byte address before writing
            byte_addr = addr * 4
            shared_memory.write(byte_addr, value)
        
        self.clock = 0

    def load_program(self, filename):
        instructions = []
        data_section = []
        spm_section = []  # New list for scratchpad memory data
        in_data = False
        in_text = False
        in_spm = False  # New flag for .data_scp section
        data_ptr = 0  # word address pointer
        spm_ptr = 0  # New pointer for scratchpad memory
        
        # Parse the assembly file
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_orig = line
                line = line.strip().split('#')[0].strip()
                if not line:
                    continue
                
                if line == ".data":
                    in_data = True
                    in_text = False
                    in_spm = False
                    continue
                elif line == ".text":
                    in_text = True
                    in_data = False
                    in_spm = False
                    continue
                elif line == ".data_scp":
                    in_spm = True
                    in_data = False
                    in_text = False
                    continue
                
                if in_spm:
                    # Process scratchpad memory section
                    if ":" in line:
                        label_part, rest = line.split(":", 1)
                        label = label_part.strip()
                        tokens = rest.strip().split()
                        if tokens and tokens[0] == ".word":
                            # Store the label in the data_labels dictionary
                            self.data_labels[label] = spm_ptr * 4  # Store label with SPM address
                            for word in tokens[1:]:
                                cleaned_word = word.replace(',', '').strip()
                                spm_section.append((spm_ptr, int(cleaned_word)))
                                spm_ptr += 1
                    else:
                        tokens = line.split()
                        if tokens and tokens[0] == ".word":
                            for word in tokens[1:]:
                                cleaned_word = word.replace(',', '').strip()
                                spm_section.append((spm_ptr, int(cleaned_word)))
                                spm_ptr += 1
                elif in_data:
                    # Process data section
                    if ":" in line:
                        label_part, rest = line.split(":", 1)
                        label = label_part.strip()
                        tokens = rest.strip().split()
                        if tokens and tokens[0] == ".word":
                            self.data_labels[label] = data_ptr * 4
                            for word in tokens[1:]:
                                cleaned_word = word.replace(',', '').strip()
                                data_section.append((data_ptr, int(cleaned_word)))
                                data_ptr += 1
                    else:
                        tokens = line.split()
                        if tokens and tokens[0] == ".word":
                            for word in tokens[1:]:
                                cleaned_word = word.replace(',', '').strip()
                                data_section.append((data_ptr, int(cleaned_word)))
                                data_ptr += 1
                elif in_text:
                    # Process text section
                    if ":" in line:
                        label_part = line.split(":", 1)[0].strip()
                        if not label_part:
                            continue
                            
                        label = label_part
                        self.text_labels[label] = len(instructions)
                        
                        if line.endswith(":"):
                            # This is just a label line with no instruction
                            pass
                        else:
                            # This has both a label and an instruction
                            instr = line.split(":", 1)[1].strip()
                            if instr:
                                instructions.append(instr)
                    else:
                        instructions.append(line)
        
        self.data_section = data_section
        
        # Write data to the global scratchpad memory (only once)
        self.global_scratchpad.clear()
        for addr, value in spm_section:
            byte_addr = addr * 4
            self.global_scratchpad.write(byte_addr, value)
        
        return instructions

    def load_cache_config(self, cache_filename):
        config = {}
        try:
            with open(cache_filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):  # Skip empty lines and comments
                        continue
                    
                    try:
                        # Split on first '=' only
                        if '=' not in line:
                            continue
                        
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Clean the value of any commas
                        value = value.replace(',', '')
                        
                        # Handle numeric values
                        if value.isdigit():
                            config[key] = int(value)
                        # Handle string values (like replacement policies)
                        elif value in ['LRU', 'FIFO']:
                            config[key] = value
                        else:
                            continue
                        
                    except Exception as e:
                        continue
                    
            # Validate required cache parameters
            required_params = [
                'l1i_size', 'l1d_size', 'l2_size',
                'l1_block_size', 'l2_block_size',
                'l1_associativity', 'l2_associativity',
                'l1_access_latency', 'l2_access_latency',
                'memory_latency', 'l1_replacement_policy', 'l2_replacement_policy'
            ]
            
            missing_params = [param for param in required_params if param not in config]
            if missing_params:
                raise ValueError(f"Missing required cache parameters: {', '.join(missing_params)}")
            
            return config
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache configuration file not found: {cache_filename}")
        except Exception as e:
            raise ValueError(f"Error loading cache configuration: {str(e)}")

    def run(self):
        cycle = 0
        max_cycles = 500000 # Prevent infinite loops
        stalled_cycles_threshold = 1000 # Number of cycles to wait after all cores appear to be done

        all_finished = False
        
        # Track the last PC for each core to detect infinite loops
        last_pcs = [-1] * len(self.cores)
        repeat_pc_counts = [0] * len(self.cores)
        no_progress_count = 0
        
        # Print initial cycle count
        print(f"Starting simulation - Cycle: {cycle}")
        
        while not all_finished and cycle < max_cycles:
            # Update the count of active cores
            active_cores = [core for core in self.cores if not core.finished]
            self.active_core_count = len(active_cores)
            
            # If all cores are finished, exit the simulation
            if not active_cores:
                break
            
            # Process SYNC barriers - hardware implementation
            barrier_cores = {}  # Dictionary to track cores at each barrier ID
            
            # First pass: collect cores waiting at sync points by barrier ID
            for core in active_cores:
                if core.sync_reached:
                    barrier_id = core.sync_count
                    if barrier_id not in barrier_cores:
                        barrier_cores[barrier_id] = []
                    barrier_cores[barrier_id].append(core)
            
            # Second pass: check if all active cores are at the same barrier
            for barrier_id, cores_at_barrier in barrier_cores.items():
                if len(cores_at_barrier) == self.active_core_count:
                    # All active cores have reached this barrier - release them
                    for core in cores_at_barrier:
                        core.sync_reached = False  # Release the core
            
            # Track if any core is making progress
            any_progress = False
            
            # Advance each core's pipeline and check for lack of progress
            for i, core in enumerate(self.cores):
                if not core.finished:
                    # Record PC before advancing
                    current_pc = core.global_pc
                    
                    # Advance pipeline
                    core.advance_pipeline()
                    
                    # Check if PC changed (making progress)
                    if current_pc != core.global_pc:
                        any_progress = True
                        repeat_pc_counts[i] = 0
                    elif current_pc == last_pcs[i]:
                        # If PC hasn't changed for multiple cycles and all pipeline stages are empty or stalled
                        repeat_pc_counts[i] += 1
                        if repeat_pc_counts[i] > 10 and core.IF is None and core.ID is None and core.EX is None and core.MEM is None and core.WB is None:
                            core.finished = True
                    
                    # Update last PC
                    last_pcs[i] = current_pc
            
            # If no core is making progress for a significant number of cycles, terminate
            if not any_progress:
                no_progress_count += 1
                if no_progress_count >= stalled_cycles_threshold:
                    break
            else:
                no_progress_count = 0
            
            # Check if all cores are finished
            all_finished = all(core.is_finished() for core in self.cores)
            
            cycle += 1
            
            # Print the current cycle count every cycle
            print(f"Cycle: {cycle}", end="\r")
            
            # For better performance, print less frequently in longer simulations
            if cycle % 1000 == 0:
                print(f"Reached cycle: {cycle}")
        
        # Store the final cycle count for performance calculations
        self.clock = cycle
        
        # Print the total number of cycles executed
        print(f"\nSimulation completed in {cycle} cycles\n")
        
        if cycle >= max_cycles:
            print("WARNING: Maximum cycle limit reached")
        
        # Print final register state
        print("Final Register State:")
        for i in range(len(self.cores)):
            print(f"Core {i} registers:")
            for j in range(32):  # Loop through all 32 registers
                if self.cores[i].registers[j] != 0:
                    print(f"  x{j}: {self.cores[i].registers[j]}")

    def is_finished(self):
        return all(core.finished for core in self.cores)

    def print_stall_summary(self):
        # Calculate IPC for each core
        ipc_values = []
        instructions_executed = []
        
        print("\n===== PERFORMANCE METRICS =====")
        
        # Print stall information for each core
        print("\n--- Stall Statistics ---")
        total_stalls = 0
        for i, core in enumerate(self.cores):
            stalls = core.stall_count
            total_stalls += stalls
            print(f"Core {i} - Stalls: {stalls}")
        print(f"Total stalls across all cores: {total_stalls}")
        
        # Calculate and print IPC (Instructions Per Cycle)
        print("\n--- IPC (Instructions Per Cycle) ---")
        total_instructions = 0
        for i, core in enumerate(self.cores):
            # Use the actual instructions completed counter
            instructions = core.instructions_completed
            total_instructions += instructions
            instructions_executed.append(instructions)
            
            # Calculate IPC
            if self.clock > 0:  # Avoid division by zero
                ipc = instructions / self.clock
                ipc_values.append(ipc)
                print(f"Core {i} - Instructions: {instructions}, IPC: {ipc:.4f}")
            else:
                ipc_values.append(0)
                print(f"Core {i} - Instructions: {instructions}, IPC: N/A (no cycles)")
        
        # Print total instructions and average IPC
        if ipc_values:
            avg_ipc = sum(ipc_values) / len(ipc_values)
            print(f"Total instructions across all cores: {total_instructions}")
            print(f"Average IPC across all cores: {avg_ipc:.4f}")
        
        # Access cache hierarchy for miss rates
        print("\n--- Cache Statistics ---")
        cache_hierarchy = self.cache_hierarchy
        
        # L1 Data Cache
        l1d_hits = cache_hierarchy.l1d.hits
        l1d_misses = cache_hierarchy.l1d.misses
        l1d_accesses = cache_hierarchy.l1d.accesses
        l1d_miss_rate = cache_hierarchy.l1d.get_miss_rate()
        l1d_hit_rate = 1 - l1d_miss_rate
        
        # L1 Instruction Cache
        l1i_hits = cache_hierarchy.l1i.hits
        l1i_misses = cache_hierarchy.l1i.misses
        l1i_accesses = cache_hierarchy.l1i.accesses
        l1i_miss_rate = cache_hierarchy.l1i.get_miss_rate()
        l1i_hit_rate = 1 - l1i_miss_rate
        
        # L2 Cache
        l2_hits = cache_hierarchy.l2.hits
        l2_misses = cache_hierarchy.l2.misses
        l2_accesses = cache_hierarchy.l2.accesses
        l2_miss_rate = cache_hierarchy.l2.get_miss_rate()
        l2_hit_rate = 1 - l2_miss_rate
        
        # Print L1 Data Cache Statistics
        print("\nL1 Data Cache:")
        print(f"  Hits: {l1d_hits}")
        print(f"  Misses: {l1d_misses}")
        print(f"  Accesses: {l1d_accesses}")
        print(f"  Hit Rate: {l1d_hit_rate:.4f} ({l1d_hit_rate*100:.2f}%)")
        print(f"  Miss Rate: {l1d_miss_rate:.4f} ({l1d_miss_rate*100:.2f}%)")
        
        # Print L1 Instruction Cache Statistics
        print("\nL1 Instruction Cache:")
        print(f"  Hits: {l1i_hits}")
        print(f"  Misses: {l1i_misses}")
        print(f"  Accesses: {l1i_accesses}")
        print(f"  Hit Rate: {l1i_hit_rate:.4f} ({l1i_hit_rate*100:.2f}%)")
        print(f"  Miss Rate: {l1i_miss_rate:.4f} ({l1i_miss_rate*100:.2f}%)")
        
        # Print L2 Cache Statistics
        print("\nL2 Cache:")
        print(f"  Hits: {l2_hits}")
        print(f"  Misses: {l2_misses}")
        print(f"  Accesses: {l2_accesses}")
        print(f"  Hit Rate: {l2_hit_rate:.4f} ({l2_hit_rate*100:.2f}%)")
        print(f"  Miss Rate: {l2_miss_rate:.4f} ({l2_miss_rate*100:.2f}%)")
        
        # Calculate and print AMAT (Average Memory Access Time)
        l1_latency = cache_hierarchy.l1d.access_latency
        l2_latency = cache_hierarchy.l2.access_latency
        memory_latency = cache_hierarchy.memory_latency
        
        amat = l1_latency + l1d_miss_rate * (l2_latency + l2_miss_rate * memory_latency)
        

    def print_memory_contents(self):
        """Display the entire shared memory contents and scratchpad memory"""
        print("\n===== MEMORY CONTENTS =====")
        
        # Get the first core's shared memory (they all share the same memory)
        if not self.cores:
            print("No cores available to access memory")
            return
            
        memory = self.cores[0].shared_memory.memory
        
        # Find non-zero memory locations to avoid printing a lot of zeros
        non_zero_locations = [(addr, val) for addr, val in enumerate(memory) if val != 0]
        
        if not non_zero_locations:
            print("All memory locations contain 0")
        else:
            # Sort by address for readability
            non_zero_locations.sort(key=lambda x: x[0])
            
            print(f"Non-zero memory locations (Total: {len(non_zero_locations)}):")
            print("Address (Byte) | Value")
            print("-" * 30)
            
            for addr, val in non_zero_locations:
                byte_addr = addr * 4  # Convert word address to byte address
                print(f"{byte_addr:12d} | {val}", end='  ')  # Print side by side

            print()  # Add a newline at the end for better formatting

        # Print labeled memory regions
        for label, byte_addr in self.data_labels.items():
            if label != 'sum':  # Already printed sum above
                word_addr = byte_addr // 4
                if 0 <= word_addr < len(memory):
                    print(f"\n{label} region (Address: {byte_addr}): Value: {memory[word_addr]}")
        
        # Print scratchpad memory contents
        print("\n===== SCRATCHPAD MEMORY CONTENTS =====")
        scratchpad = self.global_scratchpad.memory
        
        # Find non-zero scratchpad locations
        non_zero_spm = [(addr, val) for addr, val in enumerate(scratchpad) if val != 0]
        
        if not non_zero_spm:
            print("All scratchpad memory locations contain 0")
        else:
            # Sort by address for readability
            non_zero_spm.sort(key=lambda x: x[0])
            
            print(f"Non-zero scratchpad memory locations (Total: {len(non_zero_spm)}):")
            print("Address (Byte) | Value")
            print("-" * 30)
            
            for i, (addr, val) in enumerate(non_zero_spm):
                byte_addr = addr * 4  # Convert word address to byte address
                print(f"{byte_addr:12d} | {val}", end='  ')
                
                # Break line every 6 values for better readability
                if (i + 1) % 6 == 0:
                    print()
                
            print()  # Add a newline at the end for better formatting

# Main execution
if __name__ == "__main__":
    sim = PipelinedSimulator('algorithm1.asm', 'cache.txt')
    
    # Data forwarding configuration
    enabling = False
    df = input("Enter y for DATA FORWARDING, any other input for NO FORWARDING : ")
    if df.lower() == "y":
        enabling = True
    
    for core in sim.cores:
        core.data_forwarding = enabling
    
    # Run the simulation
    print(f"\nRunning simulation with DATA FORWARDING {'ENABLED' if enabling else 'DISABLED'}")
    sim.run()
    
    # Print summary information
    print("\n===== SIMULATION COMPLETED =====")
    print(f"Data forwarding was {'enabled' if enabling else 'disabled'}")
    sim.print_memory_contents()
    sim.print_stall_summary()
