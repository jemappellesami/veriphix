OPENQASM 2.0;
include "qelib1.inc";
qreg q24[5];
cx q24[4],q24[3];
cx q24[2],q24[3];
cx q24[2],q24[1];
cx q24[1],q24[0];
