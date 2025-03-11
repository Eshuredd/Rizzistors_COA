

.text

addi x20 x0 1
sw x20 0(x0)
addi x20 x0 2
sw x20 4(x0)
addi x20 x0 3
sw x20 8(x0)
addi x20 x0 4
sw x20 12(x0)
addi x20 x0 5
sw x20 16(x0)
addi x20 x0 6
sw x20 20(x0)
addi x20 x0 7
sw x20 24(x0)
addi x20 x0 8
sw x20 28(x0)
addi x20 x0 9
sw x20 32(x0)
addi x20 x0 10
sw x20 36(x0)
addi x20 x0 11
sw x20 40(x0)
addi x20 x0 12
sw x20 44(x0)
addi x20 x0 13
sw x20 48(x0)
addi x20 x0 14
sw x20 52(x0)
addi x20 x0 15
sw x20 56(x0)
addi x20 x0 16
sw x20 60(x0)
add x20 x0 x0

add x1 x31 x0 #x1 = cid
addi x2 x0 4 #array length
add x3 x0 x0

addi x4 x0 4
mul x5 x4 x2 #to be added to base = 100
mul x6 x5 x1
add x7 x3 x6 #addresss

add x10 x0 x0 #i
add x11 x0 x0 #sum

loop:
    beq x10 x2 exit
    mul x12 x10 x4
    add x13 x7 x12
    lw x14 0(x13)
    add x11 x11 x14 
    addi x10 x10 1
    j loop

final:
    
    mul x17 x4 x1
    add x18 x17 x16
    sw x11 0(x18)

bne x1 x0 exit #exit for all cores except core 0

add x21 x0 x0 #coreid
addi x22 x0 4 #4 cores
add x30 x0 x0 #final sum

loop2:
    beq x21 x22 exit2
    mul x24 x4 x21 
    add x25 x16 x24
    lw x26 0(x25)
    add x30 x30 x26
    addi x21 x21 1
    j loop2

exit2:

    sw x30 0(x28) #storing final sum
    add x10 x0 x30
    addi x17 x0 1
    

exit:
    add x0 x0 x0 #final sum will be stored in x30 of core 0