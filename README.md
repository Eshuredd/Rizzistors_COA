# COA Rizzistors Group Project 

This project is a multi-core RISC-V simulator developed as part of the course. It simulates a system with four cores capable of executing a subset of RISC-V instructions.

## Features

- Simulates a multi-core environment with four independent cores.
- Supports a subset of RISC-V instructions:
    - ADD/SUB
    - BNE
    - JAL (jump)
    - LW/SW
    - ADDI (immediate-type instruction)
- Each core has a read-only special-purpose register containing its core number (CID).
- Simulates shared memory access among the four cores.
- Implements bubble sort on all cores.
- Provides visualization of registers and memory.

## Usage

1.  **Prerequisites:**
    - Python 3.x
    - Matplotlib

2.  **Assembly File:**

    - The simulator reads an assembly file (`test.asm`) containing the RISC-V instructions.
    - The same instructions are executed on all cores.

3.  **Output:**

    - The simulator displays the contents of the registers for each core and the memory.
    - Matplotlib is used to visualize the register and memory states.

## Project Structure

-   `simulator.py`: The main simulator script containing the core logic and implementation.
-   `test.asm`: An example assembly file that the simulator can execute.
-   `README.md`: This file, providing an overview of the project.

## Code Description

The code is structured into two main classes:

*   `Cores`: Represents an individual core with its registers, program counter, and other core-specific information.
*   `Simulator`: Manages the overall simulation, including memory, cores, and program execution.

The main steps of the simulation are as follows:

1.  Load the program from the assembly file.
2.  Initialize the cores and their registers.
3.  Run the simulation, executing instructions on each core.
4.  Display the results, including register and memory contents.

## Instructions Supported

-   `ADD`: Addition of two registers
-   `SUB`: Subtraction of two registers
-   `BNE`: Branch if Not Equal
-   `JAL`: Jump and Link
-   `LW`: Load Word
-   `SW`: Store Word
-   `ADDI`: Add Immediate

## Memory Map

-   The simulator supports 4kB of memory.
-   All cores have access to the same shared memory.

## Team Members

*   Sreekar Reddy
*   Harshith Gangaraju

## Meeting Minutes

### Phase 1

**Meeting 1**

*   **Date:** January 29, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Initiated Phase 1: Build a multi-core RISC-V simulator, draw inspiration from Ripes.
    2.  The simulator to support four cores able to simulate a multi-core environment.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: Set up the basic project structure.
    *   Sreekar Reddy: Identify core RISC-V instructions to support and explore the manual.

**Meeting 2**

*   **Date:** February 5, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Defined memory map: 4kB with code read directly from the assembly file (no code in memory).
    2.  Determined supported instructions: ADD, SUB, BNE, JAL, LW, SW, ADDI.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: Implement ADD, SUB, ADDI instructions.
    *   Sreekar Reddy: Implement BNE, JAL, LW, SW instructions.

**Meeting 3**

*   **Date:** February 12, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Simulator will execute the same instructions on all cores.
    2.  Each core will have a special register for its ID (CID).
*   **Tasks Assigned:**
    *   Harshith Gangaraju: Implement a mechanism for the same instructions to all cores.
    *   Sreekar Reddy: test the code that ensures register functions such as the CID for the processore function well.

**Meeting 4**

*   **Date:** February 19, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Simulator should display the contents of registers and memory at the end of execution.
    2.  Implemented bubble sort.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: add to the code memory functions and access.
    *   Sreekar Reddy: Check and Design the all test cases so every possible instruction is being tested for bugs..

### Phase 2

**Meeting 5**

*   **Date:** February 26, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Initiate Phase 2: Extend the simulator with pipelining (and data forwarding).
    2.  The user should have an option to enable or disable forwarding.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: Research on the method to extend simulator functions.
    *   Sreekar Reddy: Look for all bugs in prior and the new functions to solve.

**Meeting 6**

*   **Date:** March 1, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Arithmetic instructions can have variable latencies. The user should be able to specify latencies for each instruction.
    2.  There will be only one fetch unit for all the compute units.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: implement the variable access that is easy to extend.
    *   Sreekar Reddy: design code around the central memory core.

**Meeting 7**

*   **Date:** March 3, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  The compute units share the instruction memory and data memory.
    2.  Compute units will have its decode/register fetch, execute, memory and writeback stages.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: test compute units access.
    *   Sreekar Reddy: Debug the units to see if it can execute code.

**Meeting 8**

*   **Date:** March 6, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  Implemented array addition for all compute units.
    2.  The array addition would compute a different portion in each unit, and total to a number at the end.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: implement various test cases to verify array functions.
    *   Sreekar Reddy: check simulator for test cases and any problems to debug.

**Meeting 9**

*   **Date:** March 10, 2025
*   **Members:** Harshith Gangaraju, Sreekar Reddy
*   **Decisions:**
    1.  The project is complete.
    2.  Push the code to GitHub and make all documentation.
*   **Tasks Assigned:**
    *   Harshith Gangaraju: update readme.
    *   Sreekar Reddy: Finalise the codes and push to the hub.

## RISC-V Simulator
- This is a project to simulate assembly code files in python and can debug and test the files without needing any physical hardware
- This simulator can be used to verify correctness
---
## Features
-  Simulates 4 cores running concurrently.
-  Loads and processes assembly files with .data, .text, and .word directives. Data labels are mapped to memory addresses, and instructions reference these labels.
-  Implements arithmetic operations (ADD, SUB, ADDI), branching (BLT, BNE), memory operations (LW, SW), and jumps (J, JAL, JALR).
-  We can observe the stages one step at a time to observe the state of registers
-  We can handle errors much better in this simulator than normal assembly code
-  For Visualization we use MatPlotLib to make plots too show registers and memory
---
## Requirements

- Python 3.x
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
---
## How to Use
1. Have a assembly file named BubbleSort.asm 

2. type the command python project3.py in the terminal
---

