# COA Rizzistors Group Project 

## Members are :
- 1.Sreekar Reddy
- 2.Gangaraju harshith

## RISC-V Simulator
- This is a project to simulate assembly code files in python and can debug and test the files without needing any physical hardware
- This simulator can be used to verify correctness
---
## Features
-  Simulates 4 cores running concurrently.
-  Loads and processes assembly files with .data, .text, and .word directives. Data labels are mapped to memory addresses, and instructions reference these labels.
- Implements arithmetic operations (ADD, SUB, ADDI), branching (BLT, BNE), memory operations (LW, SW), and jumps (J, JAL, JALR).
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
### MINUTES OF THE MEETING :
#### 1ST MEETING
- Decisions : Majority of the people were choosing either c++ or python so we did some research and finally decided on Python

#### 2ND MEETING
- Decisions: Sreekar worked on basic Instructions while Harshith worked on file parsing and reading

#### 3RD MEETING
- Decisions: Sreekar worked on loading and storing the words Harshith worked on Bubble sort

#### 4TH MEETING
- Decisions: There was a problem with loading words from memory when using .data and .text so we rectified it




