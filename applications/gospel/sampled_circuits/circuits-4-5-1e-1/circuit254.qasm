OPENQASM 2.0;
include "qelib1.inc";
qreg q255[4];
cx q255[2],q255[1];
cx q255[3],q255[2];
cx q255[2],q255[1];
cx q255[1],q255[0];
