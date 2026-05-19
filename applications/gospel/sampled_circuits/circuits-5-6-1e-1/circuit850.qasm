OPENQASM 2.0;
include "qelib1.inc";
qreg q851[5];
cx q851[3],q851[4];
cx q851[2],q851[3];
cx q851[1],q851[2];
cx q851[0],q851[1];
rx(pi/4) q851[1];
