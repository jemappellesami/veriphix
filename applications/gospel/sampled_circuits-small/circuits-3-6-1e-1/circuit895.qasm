OPENQASM 2.0;
include "qelib1.inc";
qreg q896[3];
cx q896[0],q896[1];
rx(pi) q896[2];
cx q896[1],q896[2];
cx q896[0],q896[1];
rx(pi/4) q896[1];
