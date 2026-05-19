OPENQASM 2.0;
include "qelib1.inc";
qreg q86[5];
cx q86[3],q86[4];
cx q86[3],q86[2];
cx q86[2],q86[1];
cx q86[0],q86[1];
rx(pi/4) q86[1];
